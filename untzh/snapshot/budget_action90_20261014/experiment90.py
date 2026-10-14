import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, '/data/lc/untzh/risk_route89_20261014')
import experiment89 as route
core, base = route.core, route.base
ROOT=Path(__file__).resolve().parent
route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=ROOT
N, TOTAL, SIZE=route.N, route.TOTAL, route.SIZE
SPECS={f'{signal}_{policy}':(signal,policy) for signal in ['absolute128','absolute_full','violation_full','budget_full'] for policy in ['uniform','tree']}
route.ARMS=list(SPECS)


def extra_data(truths, data):
    tick=time.monotonic();c=base.load(route.SOURCE/'cache/cohort.npz')
    legal=np.ones((SIZE,TOTAL-N),dtype=bool)
    for i,(h,r,g) in enumerate(c['queries']):
        ids=[x-N for x in truths[int(h),int(r)] if x>=N]
        if ids:legal[i,ids]=False
    data['legal']=torch.as_tensor(legal,device='cuda')
    data['capacity']=torch.as_tensor(10-data['fixed'][:,0].astype(np.int64),device='cuda')
    data['qnorm']=data['q'].norm(dim=-1).clamp_min(1e-30)
    data['selections']=base.load(route.SOURCE/'cache/protection.npz')['selections']
    assert ((data['capacity']>=0)&(data['capacity']<=9)).all()
    core.dump('extra-data.json',dict(seconds=time.monotonic()-tick,legal_bytes=legal.nbytes,full_candidates=TOTAL-N))
    return time.monotonic()-tick


def penalty(E,data,chosen,signal):
    if signal=='absolute128':
        p=core.penalty(E,data,chosen,'absolute')
        return p,dict(active_queries=len(chosen),active_pairs=len(chosen)*128)
    q=data['q'][chosen];scores=q@E[N:].double().T;legal=data['legal'][chosen]
    if signal=='absolute_full':
        values=F.softplus(scores).masked_fill(~legal,0)
        p=(values.sum(-1)/legal.sum(-1)).mean()
        return p,dict(active_queries=len(chosen),active_pairs=int(legal.sum()))
    gap=(scores-data['gold'][chosen,None])/data['qnorm'][chosen,None]
    active=(gap.detach()>0)&legal
    if signal=='budget_full':
        # Allow at most capacity new outrankers. Keep the largest positive
        # gaps; penalize the smallest excess gaps, a score-space surrogate.
        allowance=data['capacity'][chosen]
        top=gap.detach().masked_fill(~legal,-torch.inf).topk(9,dim=-1).indices
        allowed=torch.arange(9,device='cuda')[None,:]<allowance[:,None]
        keep=torch.zeros_like(active)
        keep.scatter_(1,top,allowed)
        active=active&~keep
    values=F.relu(gap).square().masked_fill(~active,0)
    p=(values.sum(-1)/legal.sum(-1)).mean()
    return p,dict(active_queries=int(active.any(-1).sum()),active_pairs=int(active.sum()))


def train(name,spec,initial,prefix,new,tape,data,model):
    signal,policy=spec;tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix)
    rng=np.random.default_rng(890029);steps=[];decisions=[];selection_seconds=0.
    for j in range(220,440):
        ep=int(tape['epochs'][j])-1
        if policy=='tree' and (j==220 or tape['epochs'][j-1]!=tape['epochs'][j]):
            torch.cuda.synchronize();start=time.monotonic()
            X=route.features(E,data,ep);scores=model.predict_proba(X)[:,1]
            pool=np.argsort(-scores,kind='stable')[:int(np.ceil(.05*SIZE))]
            torch.cuda.synchronize();seconds=time.monotonic()-start;selection_seconds+=seconds
            decisions.append(dict(epoch=ep,seconds=seconds,pool=pool.tolist()))
        if policy=='uniform':chosen=data['selections'][j-220]
        else:
            first=rng.choice(pool,26,replace=False);available=np.ones(SIZE,bool);available[first]=False
            chosen=np.concatenate([first,rng.choice(np.flatnonzero(available),6,replace=False)])
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,stats=penalty(E,data,chosen,signal);gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and np.isfinite([bn,pn,alpha]).all()
        combined=gl+alpha*gp;assert torch.isfinite(combined).all()
        opt.zero_grad(set_to_none=True);E.grad=combined;opt.step();assert torch.isfinite(E).all()
        steps.append(dict(step=j,epoch=ep+1,penalty=float(p),alpha=alpha,gradient_ratio=alpha*pn/bn if bn else 0.,**stats))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:
            core.progress('train',arm=name,epoch=ep+1,active_pairs=stats['active_pairs'])
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',steps);base.write(out/'decisions.json',decisions)
    result=dict(seconds=time.monotonic()-tick,selection_seconds_included=selection_seconds,steps=220,
                active_steps=sum(x['active_pairs']>0 for x in steps),mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in steps])),
                mean_active_queries=float(np.mean([x['active_queries'] for x in steps])),mean_active_pairs=float(np.mean([x['active_pairs'] for x in steps])),
                max_gradient_ratio=max(x['gradient_ratio'] for x in steps),frozen_equal=True)
    assert result['max_gradient_ratio']<=.100001
    base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare()
    es=extra_data(truths,data)
    costs=dict(preparation_seconds=ps+es,source=str(route.SOURCE),normalization='rho=.1 each nonzero new-row protection gradient; no fixed-lambda cap',training={})
    for name,spec in SPECS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,model)
        core.dump('costs-running.json',costs)
    metrics,seconds=route.evaluate(truths);costs.update(evaluation_seconds=seconds,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
