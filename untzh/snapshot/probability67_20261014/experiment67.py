"""Candidate probability-mass protection and matched learning-objective controls."""
import collections
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,'/data/lc/untzh/search64_20261013')
import experiment64 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 'host_mass_old':dict(host='original',signal='mass_old'),
 'host_mass_gold':dict(host='original',signal='mass_gold'),
 'host_mass_guard':dict(host='original',signal='mass_guard'),
 'host_mass_sample':dict(host='original',signal='mass_sample'),
 **{f'{host}_{signal}':dict(host=host,signal=signal)
     for host in ['contrastive','fullsoftmax'] for signal in ['none','absolute','semantic','mass_old']},
}


def prepare(initial,old,new,tape):
    data=prev.enrich(core.prepare_data(initial,old,new),new,len(initial['R']))
    truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    e=initial['E'][:ex.N].cuda().double();logs=[]
    with torch.no_grad():
        for lo in range(0,ex.SIZE,128):
            hi=min(lo+128,ex.SIZE);scores=data['q_t'][lo:hi]@e.T
            rows=[];cols=[]
            for i,(h,r,t) in enumerate(data['queries'][lo:hi]):
                for c in truth[int(h),int(r)]:
                    if c<ex.N and c!=int(t):rows.append(i);cols.append(c)
            scores[rows,cols]=-torch.inf
            logs.append(torch.logsumexp(scores,dim=-1))
    data['old_logZ_t']=torch.cat(logs)
    # Reuse masks for repeated epochs; only known TRAIN positives are removed.
    ce_masks={}
    for j in range(370,740):
        lo,hi=tape['bounds'][j:j+2];p=new[tape['pos'][lo:hi]]
        rows=[];cols=[]
        for i,(h,r,t) in enumerate(p):
            for c in truth[int(h),int(r)]:
                if c!=int(t):rows.append(i);cols.append(c)
        ce_masks[j]=(torch.as_tensor(rows,device='cuda',dtype=torch.long),torch.as_tensor(cols,device='cuda',dtype=torch.long))
    data['ce_masks']=ce_masks
    ex.dump('probability-preparation.json',{'old_logZ_range':[float(data['old_logZ_t'].min()),float(data['old_logZ_t'].max())],
        'ce_mask_pairs':sum(len(v[0]) for v in ce_masks.values()),'source':'TRAIN-only truths; other old answers removed except current gold'})
    return data


def learning(E,R,new,tape,j,host,data):
    if host=='original':return ex.base.learning(E,R,new,tape,j)
    lo,hi=tape['bounds'][j:j+2]
    p=torch.as_tensor(new[tape['pos'][lo:hi]],device='cuda');h,r,t=p.unbind(1)
    if host=='fullsoftmax':
        logits=(E[h]*R[r])@E.T
        rows,cols=data['ce_masks'][j]
        logits=logits.index_put((rows,cols),torch.full_like(rows,-torch.inf,dtype=logits.dtype))
        loss=F.cross_entropy(logits,t)
    elif host=='contrastive':
        nh=torch.as_tensor(tape['nh'][lo:hi],device='cuda');nt=torch.as_tensor(tape['nt'][lo:hi],device='cuda')
        ps=(E[h]*R[r]*E[t]).sum(-1)
        ns=(E[nh]*R[r[:,None]]*E[nt]).sum(-1)
        loss=F.cross_entropy(torch.cat([ps[:,None],ns],dim=1),torch.zeros(len(p),dtype=torch.long,device='cuda'))
    else:raise ValueError(host)
    return loss+.0005*(E[h].square().mean()+R[r].square().mean())


def penalty(E,data,selected,signal):
    if signal in ['absolute','semantic']:
        return prev.penalty(E,data,selected,'absolute' if signal=='absolute' else 'relation_soft')
    q=data['q_t'][selected]
    if signal=='mass_sample':
        ids=data['candidates_t'][selected]
        scores=(q[:,None]*E[ids].double()).sum(-1)
        lsenew=torch.logsumexp(scores,dim=-1)+data['legal_t'][selected].sum(-1).double().log()-np.log(128)
    else:
        scores=(q@E[ex.N:].double().T).masked_fill(~data['legal_t'][selected],-torch.inf)
        lsenew=torch.logsumexp(scores,dim=-1)
    ref=data['gold_t'][selected] if signal=='mass_gold' else data['old_logZ_t'][selected]
    each=F.softplus(lsenew-ref)
    if signal=='mass_guard':each=each+.1*F.softplus(lsenew-data['gold_t'][selected])
    return each.mean()


def calibrate(initial,prefix,new,tape,data):
    report={};rng=np.random.default_rng(17)
    selections=[rng.choice(ex.SIZE,32,replace=False) for _ in range(8)]
    for name,spec in ARMS.items():
        if spec['signal']=='none':continue
        E,R,opt=core.model(initial,prefix);rows=[]
        for idx,j in enumerate(range(370,378)):
            loss=learning(E,R,new,tape,j,spec['host'],data);gl=torch.autograd.grad(loss,E)[0]
            p=penalty(E,data,selections[idx],spec['signal']);gp=torch.autograd.grad(p,E)[0]
            rows.append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
        a=np.array(rows);denom=np.sqrt(np.mean(a[:,1]**2))
        report[name]={'lambda':.1*np.sqrt(np.mean(a[:,0]**2))/denom if denom else 0.,'norms':rows}
    ex.dump('calibration.json',report);return report


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    for j in range(370,740):
        selected=rng.choice(ex.SIZE,32,replace=False)
        loss=learning(E,R,new,tape,j,spec['host'],data);gl=torch.autograd.grad(loss,E)[0]
        p=E.new_zeros(());alpha=0.;gp=torch.zeros_like(E)
        if spec['signal']!='none':
            p=penalty(E,data,selected,spec['signal']);gp=torch.autograd.grad(p,E)[0]
            bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
            alpha=min(cal[name]['lambda'],.1*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'frozen_rows_equal':True}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();data=prepare(initial,old,new,tape);cal=calibrate(initial,prefix,new,tape,data)
    costs={'prepare_seconds':time.monotonic()-tick,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
