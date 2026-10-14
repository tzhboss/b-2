"""Differential-rate old-model adaptation with replay/KL controls."""
import collections
import copy
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
 **{f'{permission}_{host}_{memory}':(permission,host,memory)
     for permission in ['R','old64R','allR'] for host in ['original','mixed'] for memory in ['none','replay']},
 **{f'{permission}_original_klmass':(permission,'original','klmass') for permission in ['R','old64R','allR']},
}


def model(prefix,ids):
    old=prefix['E'][:ex.N].cuda().clone()
    newp=torch.nn.Parameter(prefix['E'][ex.N:].cuda().clone())
    oldp=torch.nn.Parameter(old[ids].clone()) if len(ids) else None
    R=torch.nn.Parameter(prefix['R'].cuda().clone())
    on=torch.optim.Adam([newp],lr=.007)
    saved=next(iter(prefix['optimizer']['state'].values()))
    on.state[newp]={k:v[ex.N:].cuda().clone() if k in ['exp_avg','exp_avg_sq'] else copy.deepcopy(v) for k,v in saved.items()}
    oo=torch.optim.Adam([oldp],lr=.000007) if oldp is not None else None
    orr=torch.optim.Adam([R],lr=.00007)
    params=[newp]+([oldp] if oldp is not None else [])+[R]
    opts=[on]+([oo] if oo is not None else [])+[orr]
    def entity():return torch.cat([old.index_copy(0,ids,oldp) if oldp is not None else old,newp])
    return params,opts,R,entity


def masks(queries,truth,total,old_only=False):
    legal=np.ones((len(queries),total),bool)
    for i,(h,r,t) in enumerate(queries):
        for c in truth[int(h),int(r)]:
            if c<total and c!=int(t):legal[i,int(c)]=False
    return torch.as_tensor(legal,device='cuda')


def memory(E,R,queries,teacherE,teacherR,legal,kind):
    p=torch.as_tensor(queries,device='cuda');h,r,t=p.unbind(1)
    scores=(E[h]*R[r])@E.T
    scores=scores.masked_fill(~legal,-torch.inf)
    if kind=='replay':return F.cross_entropy(scores,t)
    teacher_scores=((teacherE[h]*teacherR[r])@teacherE[:ex.N].T).masked_fill(~legal[:,:ex.N],-torch.inf)
    logp=F.log_softmax(scores[:,:ex.N],dim=-1)
    teacherp=F.softmax(teacher_scores,dim=-1).detach()
    # Avoid 0*(-inf) on filtered coordinates.
    kl=(teacherp*(teacherp.clamp_min(1e-30).log()-logp.masked_fill(~legal[:,:ex.N],0))).sum(-1).mean()
    newmass=F.softplus(torch.logsumexp(scores[:,ex.N:],dim=-1)-torch.logsumexp(scores[:,:ex.N],dim=-1)).mean()
    return kl+newmass


def learning(E,R,new,tape,j,host):
    if host=='original':return core.learning(E,R,new,tape,j)
    lo,hi=tape['bounds'][j:j+2]
    p,nh,nt=[torch.as_tensor(a,device='cuda') for a in (new[tape['pos'][lo:hi]],tape['nh'][lo:hi],tape['nt'][lo:hi])]
    h,r,t=p.unbind(1);ps=(E[h]*R[r]*E[t]).sum(-1);ns=(E[nh]*R[r[:,None]]*E[nt]).sum(-1)
    ordinary=F.softplus(-ps).mean()+F.softplus(ns).mean()
    ce=F.cross_entropy(torch.cat([ps[:,None],ns],dim=1),torch.zeros(len(p),dtype=torch.long,device='cuda'))
    return .5*(ordinary+ce)+.0005*(E[h].square().mean()+R[r].square().mean())


def gradnorm(grads):return np.sqrt(sum(float(g.double().square().sum()) for g in grads))


def train(name,spec,initial,prefix,new,tape,data,truth,ids):
    started=time.monotonic();permission,host,kind=spec
    params,opts,R,entity=model(prefix,ids)
    teacherE=prefix['E'].cuda();teacherR=prefix['R'].cuda()
    rng=np.random.default_rng(17);selected=[rng.choice(ex.SIZE,32,replace=False) for _ in range(370)]
    coefficient=0.
    if kind=='klmass':
        norms=[]
        for j in range(370,378):
            E=entity();loss=learning(E,R,new,tape,j,host);gl=torch.autograd.grad(loss,params)
            E=entity()
            queries=data['queries'][selected[j-370]];legal=masks(queries,truth,ex.TOTAL)
            pen=memory(E,R,queries,teacherE,teacherR,legal,kind);gp=torch.autograd.grad(pen,params)
            norms.append([gradnorm(gl),gradnorm(gp)])
        a=np.array(norms);coefficient=.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2))
    records=[]
    for j in range(370,740):
        E=entity();loss=learning(E,R,new,tape,j,host)
        queries=data['queries'][selected[j-370]];pen=E.new_zeros(());alpha=0.
        if kind=='replay':
            pen=memory(E,R,queries,teacherE,teacherR,masks(queries,truth,ex.TOTAL),kind)
            gl=torch.autograd.grad(loss+.1*pen,params)
            alpha=.1
        else:
            gl=torch.autograd.grad(loss,params)
            if kind=='klmass':
                E=entity()
                pen=memory(E,R,queries,teacherE,teacherR,masks(queries,truth,ex.TOTAL),kind)
                gp=torch.autograd.grad(pen,params);bn,pn=gradnorm(gl),gradnorm(gp)
                alpha=min(coefficient,.1*bn/pn) if pn else 0.
                gl=[a+alpha*b for a,b in zip(gl,gp)]
        if not torch.isfinite(loss) or not torch.isfinite(pen) or not all(torch.isfinite(g).all() for g in gl):raise FloatingPointError(name)
        for opt in opts:opt.zero_grad(set_to_none=True)
        for p,g in zip(params,gl):p.grad=g
        for opt in opts:opt.step()
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'new_loss':float(loss),'memory_loss':float(pen),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    E=entity().detach().cpu();fixed=torch.ones(ex.N,dtype=torch.bool);fixed[ids.cpu()]=False
    assert torch.equal(E[:ex.N][fixed],prefix['E'][:ex.N][fixed]) and torch.isfinite(E).all() and torch.isfinite(R).all()
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.double(),'R':R.detach().cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'old_E_lr':.000007,'R_lr':.00007,'new_E_lr':.007,
            'old_E_drift':float((E[:ex.N]-prefix['E'][:ex.N]).double().norm()),
            'R_drift':float((R.detach().cpu()-prefix['R']).double().norm()),'lambda':coefficient,'unopened_old_rows_equal':True}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    data=ex.base.load(core.PRIOR/'cache/cohort.npz')
    truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    ends=np.concatenate([new[:,0],new[:,2]]);count=np.bincount(ends[ends<ex.N],minlength=ex.N)
    best=np.lexsort((np.arange(ex.N),-count))[:64]
    permissions={'R':torch.empty(0,dtype=torch.long,device='cuda'),
                 'old64R':torch.as_tensor(best,device='cuda'),'allR':torch.arange(ex.N,device='cuda')}
    ex.dump('permissions.json',{'old64':best.tolist(),'old_E_lr':.000007,'R_lr':.00007,'new_E_lr':.007})
    costs={'training':{}}
    for name,spec in ARMS.items():
        saved=ROOT/'models'/name/'training.json'
        if saved.exists():
            costs['training'][name]=json.loads(saved.read_text())
            ex.progress('reuse_completed_arm',arm=name)
            continue
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,truth,permissions[spec[0]]);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
