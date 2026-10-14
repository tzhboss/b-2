"""Protection population and batch-variance study with cached TRAIN sampling."""
import collections
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,'/data/lc/untzh/search63_20261013')
import experiment63 as core
ex=core.ex
ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 **{f'cohort_uniform{batch}':('cohort','uniform',batch) for batch in [32,128]},
 **{f'global_uniform{batch}':('global','uniform',batch) for batch in [32,128,512]},
 **{f'global_{route}{batch}':('global',route,batch) for route in ['query','relation','boundary'] for batch in [32,128]},
 'global_semantic128':('global','semantic',128),
}


def prepare(initial,old,new):
    started=time.monotonic()
    A=ex.base.load(ex.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    eligible=np.flatnonzero(A['old_train'][:,1]<=10)
    queries=old[eligible]
    q=initial['E'][queries[:,0]].double()*initial['R'][queries[:,1]].double()
    known=collections.defaultdict(set)
    for h,r,t in new:
        if t>=ex.N:known[int(h),int(r)].add(int(t))
    rng=np.random.default_rng(17)
    candidates=rng.integers(ex.N,ex.TOTAL,size=(len(eligible),128),dtype=np.int32)
    for i,(h,r,t) in enumerate(queries):
        legalbad=known[int(h),int(r)]
        if legalbad:
            bad=np.isin(candidates[i],list(legalbad))
            while bad.any():
                candidates[i,bad]=rng.integers(ex.N,ex.TOTAL,size=int(bad.sum()))
                bad=np.isin(candidates[i],list(legalbad))
    cohort=ex.base.load(core.PRIOR/'cache/cohort.npz')
    positions=np.searchsorted(eligible,cohort['ids'])
    candidates[positions]=cohort['candidates']
    group=np.unique(queries[:,:2],axis=0,return_inverse=True)[1]
    groupcount=np.bincount(group)
    relationcount=np.bincount(queries[:,1],minlength=len(initial['R']))
    qprob=1/groupcount[group];qprob/=qprob.sum()
    rprob=1/relationcount[queries[:,1]];rprob/=rprob.sum()
    semantic=np.array([(int(h),int(r)) not in known or not known[int(h),int(r)] for h,r,t in queries])
    data={'q':q.cuda(),'candidates':torch.as_tensor(candidates,device='cuda',dtype=torch.long),
          'queries':queries,'eligible':eligible,'cohort':positions,'query_prob':qprob,'relation_prob':rprob,
          'boundary':np.argsort(-A['old_train'][eligible,1],kind='stable')[:int(np.ceil(.2*len(eligible)))],
          'semantic':np.flatnonzero(semantic)}
    np.savez(ROOT/'population.npz',eligible=eligible,queries=queries,cohort_positions=positions,group=group,
             query_prob=qprob,relation_prob=rprob,semantic=semantic)
    report={'seconds':time.monotonic()-started,'eligible_answers':len(eligible),'unique_queries':int(group.max()+1),
            'relations':int((relationcount>0).sum()),'old_only_answers':int(semantic.sum()),
            'candidate_draws':128,'q_bytes':q.numel()*q.element_size(),'candidate_bytes':candidates.nbytes,
            'cohort_candidates_reused_exact':True}
    ex.dump('preparation.json',report)
    return data,report


def selections(data,spec,rng):
    population,route,batch=spec
    n=len(data['eligible'])
    if population=='cohort':return rng.choice(data['cohort'],batch,replace=False)
    if route=='uniform':return rng.choice(n,batch,replace=False)
    if route in ['query','relation']:return rng.choice(n,batch,replace=False,p=data[route+'_prob'])
    if route=='semantic':return rng.choice(data['semantic'],batch,replace=False)
    if route=='boundary':
        targeted=int(round(batch*.8125))
        return np.concatenate([rng.choice(data['boundary'],targeted,replace=False),rng.choice(n,batch-targeted,replace=False)])
    raise ValueError(route)


def penalty(E,data,selected):
    q=data['q'][selected];ids=data['candidates'][selected]
    return F.softplus((q[:,None]*E[ids].double()).sum(-1)).mean()


def calibration(initial,prefix,new,tape,data,spec):
    E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);rows=[]
    for j in range(370,378):
        selected=selections(data,spec,rng)
        gl=torch.autograd.grad(ex.base.learning(E,R,new,tape,j),E)[0]
        gp=torch.autograd.grad(penalty(E,data,selected),E)[0]
        rows.append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
    a=np.array(rows)
    return {'lambda':.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2)),'norms':rows}


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[];seen=set()
    for j in range(370,740):
        selected=selections(data,spec,rng);seen.update(map(int,selected))
        loss=ex.base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p=penalty(E,data,selected);gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
        alpha=min(cal['lambda'],.1*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'distinct_answers_selected':len(seen),
            'distinct_queries_selected':len(set(map(tuple,data['queries'][sorted(seen),:2]))),'frozen_rows_equal':True}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    data,prep=prepare(initial,old,new)
    cal={name:calibration(initial,prefix,new,tape,data,spec) for name,spec in ARMS.items()}
    ex.dump('calibration.json',cal);costs={'preparation':prep,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal[name]);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
