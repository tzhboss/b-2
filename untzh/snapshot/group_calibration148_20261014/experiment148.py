import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/forecast_protect146_20261014')
import experiment146 as previous
base,core=previous.base,previous.core
ROOT=Path(__file__).resolve().parent
N,TOTAL=previous.N,previous.TOTAL
ARMS=['raw','global_group','context_group','shuffled_context']
NR=474

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def sync():torch.cuda.synchronize();return time.monotonic()

def fit(truths):
    tick=time.monotonic()
    keys=np.array(sorted(truths))
    old=np.array([sum(t<N for t in truths[int(h),int(r)]) for h,r in keys])
    new=np.array([sum(t>=N for t in truths[int(h),int(r)]) for h,r in keys])
    p0=float(new.sum()/(old+new).sum())
    p=(new+2*p0)/(old+new+2)
    shuffled=p.copy()
    rng=np.random.default_rng(1480017)
    for mask in [keys[:,0]<N,keys[:,0]>=N]:shuffled[mask]=rng.permutation(p[mask])
    payload=dict(keys=keys,encoded=keys[:,0]*NR+keys[:,1],old_count=old,new_count=new,p=p,shuffled=shuffled,p0=np.array(p0))
    np.savez(ROOT/'frozen-priors.npz',**payload)
    report=dict(seconds=time.monotonic()-tick,query_keys=len(keys),old_answers=int(old.sum()),new_answers=int(new.sum()),p0=p0,
                shrinkage_mass=2,train_only=True,all_priors_before_valid=True)
    base.write(ROOT/'priors-frozen.json',report)
    return payload,report

def priors(facts,params):
    key=facts[:,0]*NR+facts[:,1]
    pos=np.searchsorted(params['encoded'],key)
    found=(pos<len(params['encoded']))
    safe=np.minimum(pos,len(params['encoded'])-1)
    found&=params['encoded'][safe]==key
    p0=float(params['p0'])
    p=np.full(len(key),p0)
    shuffled=p.copy()
    p[found]=params['p'][safe[found]]
    shuffled[found]=params['shuffled'][safe[found]]
    return np.stack([np.full(len(key),p0),p,shuffled]),found

@torch.no_grad()
def old_logz(E,R,queries):
    tick=sync()
    keys=np.unique(np.concatenate([q[q[:,0]<N,:2] for q in queries.values()]),axis=0)
    parts=[]
    for lo in range(0,len(keys),128):
        b=keys[lo:lo+128]
        q=E[b[:,0]]*R[b[:,1]]
        parts.append(torch.logsumexp(q@E[:N].T,1).cpu().numpy())
    value=np.concatenate(parts)
    np.savez(ROOT/'old-context-logz.npz',keys=keys,encoded=keys[:,0]*NR+keys[:,1],logz=value)
    return dict(encoded=keys[:,0]*NR+keys[:,1],logz=value),dict(seconds=sync()-tick,unique_old_contexts=len(keys),computed_once=True)

@torch.no_grad()
def score_old(E,R,facts,teacher,cache,truths,params,logz,split):
    tick=sync()
    p,found=priors(facts,params)
    logits=np.log(p/(1-p))
    parts=[[] for _ in range(4)]
    shifts=[]
    for lo in range(0,len(facts),128):
        b=facts[lo:lo+128]
        q=E[b[:,0]]*R[b[:,1]]
        new_scores=q@E[N:].T
        gold=cache['gold'][lo:lo+len(b)]
        ix=np.searchsorted(logz['encoded'],b[:,0]*NR+b[:,1])
        assert np.array_equal(logz['encoded'][ix],b[:,0]*NR+b[:,1])
        dz=torch.as_tensor(logz['logz'][ix],device='cuda')-torch.logsumexp(new_scores,1)
        delta=torch.as_tensor(logits[:,lo:lo+len(b)],device='cuda')+dz[None]
        shifts.append(delta.cpu().numpy())
        legal=torch.ones_like(new_scores,dtype=torch.bool)
        for i,(h,r,g) in enumerate(b):legal[i,[t-N for t in truths[int(h),int(r)] if t>=N]]=False
        for a in range(4):
            ss=new_scores if a==0 else new_scores+delta[a-1,:,None]
            above=ss>gold[:,None]
            raw=teacher[lo:lo+len(b),0]+above.sum(1).cpu().numpy()
            filtered=teacher[lo:lo+len(b),1]+(above&legal).sum(1).cpu().numpy()
            parts[a].append(np.column_stack([raw,filtered]))
    return {a:np.concatenate(part) for a,part in zip(ARMS,parts)},dict(seconds=sync()-tick,n=len(facts),train_context_covered=float(found.mean()),
        delta_quantiles={a:{str(t):float(np.quantile(np.concatenate(shifts,axis=1)[i],t)) for t in [0,.1,.5,.9,1]} for i,a in enumerate(ARMS[1:])})

@torch.no_grad()
def score_new(E,R,facts,truths,params,logz):
    tick=sync()
    p,found=priors(facts,params)
    logits=np.log(p/(1-p))
    parts=[[] for _ in range(4)]
    for lo in range(0,len(facts),64):
        b=facts[lo:lo+64]
        q=E[b[:,0]]*R[b[:,1]]
        scores=q@E.T
        oldz=torch.logsumexp(scores[:,:N],1)
        oldhead=b[:,0]<N
        if oldhead.any():
            key=b[oldhead,0]*NR+b[oldhead,1]
            ix=np.searchsorted(logz['encoded'],key)
            oldz[torch.as_tensor(oldhead,device='cuda')]=torch.as_tensor(logz['logz'][ix],device='cuda')
        delta=torch.as_tensor(logits[:,lo:lo+len(b)],device='cuda')+oldz[None]-torch.logsumexp(scores[:,N:],1)[None]
        legal=torch.ones_like(scores,dtype=torch.bool)
        for i,(h,r,g) in enumerate(b):
            legal[i,list(truths[int(h),int(r)])]=False
            legal[i,int(g)]=True
        idx=torch.as_tensor(b[:,2],device='cuda')
        row=torch.arange(len(b),device='cuda')
        for a in range(4):
            ss=scores.clone()
            if a:ss[:,N:]+=delta[a-1,:,None]
            gold=ss[row,idx].clone()
            above=ss>gold[:,None]
            parts[a].append(torch.stack([1+above.sum(1),1+(above&legal).sum(1)],1).cpu().numpy())
    return {a:np.concatenate(part) for a,part in zip(ARMS,parts)},dict(seconds=sync()-tick,n=len(facts),train_context_covered=float(found.mean()))

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    status('train_priors')
    _,_,_,_,_,truths=previous.risk.training_assets(17)
    params,fitcost=fit(truths)
    model=torch.load('/data/lc/untzh/train_init121_20261014/seed17/models/zero/epoch-20.pt',map_location='cpu',weights_only=True)
    E,R=model['E'].cuda().double(),model['R'].cuda().double()
    status('evaluation_prepare')
    fast=previous.controls.pair.fastmodule.FastOld(17,truths)
    queries=dict(fast.queries,new_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['new_valid'])
    baseline=base.load('/data/lc/untzh/train_init121_20261014/seed17/evaluation/zero.npz')
    status('old_context_logz')
    logz,lzcost=old_logz(E,R,queries)
    results={a:{} for a in ARMS}
    costs={}
    for split in ['old_train','old_valid']:
        status('score',split=split)
        ranks,costs[split]=score_old(E,R,queries[split],fast.teacher[split],fast.cache[split],truths,params,logz,split)
        assert np.array_equal(ranks['raw'],baseline[split]), 'zero-offset rank identity'
        for arm in ARMS:results[arm][split]=ranks[arm]
    status('score',split='new_valid')
    ranks,costs['new_valid']=score_new(E,R,queries['new_valid'],truths,params,logz)
    assert np.array_equal(ranks['raw'],baseline['new_valid']), 'zero-offset new rank identity'
    for arm in ARMS:results[arm]['new_valid']=ranks[arm]
    metrics={}
    drift={}
    (ROOT/'evaluation').mkdir(exist_ok=True)
    for arm in ARMS:
        states=results[arm]
        np.savez(ROOT/'evaluation'/f'{arm}.npz',**states)
        m={s:base.metrics(states[s],q) for s,q in queries.items()}
        for s in ['old_train','old_valid']:m[s]['drops']=core.drops(states[s],fast.teacher[s])
        m['directional_ratios_to_raw']=[m['new_valid'][d]['filtered_train_truths']['mrr']/metrics['raw']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS] if arm!='raw' else [1.,1.,1.]
        metrics[arm]=m
        drift[arm]={s:previous.drift(states[s],fast.teacher[s]) for s in ['old_train','old_valid']}
        print('GROUP148',arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'],m['directional_ratios_to_raw'],flush=True)
    control_report={}
    for name,path in [('fixed_anchor','/data/lc/untzh/smallinit_protect126_20261014/seed17/zero/evaluation/anchor_null.npz'),
                      ('fixed_replay','/data/lc/untzh/forecast_protect146_20261014/seed17/evaluation/fixed_replay.npz')]:
        states=base.load(path)
        m={s:base.metrics(states[s],q) for s,q in queries.items()}
        for s in ['old_train','old_valid']:m[s]['drops']=core.drops(states[s],fast.teacher[s])
        control_report[name]=m
    def values(m):return (m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'])
    candidate=values(metrics['context_group'])
    raw=values(metrics['raw'])
    anchor=values(control_report['fixed_anchor'])
    checks=dict(new_floor=candidate[0]>=raw[0]*.98,direction_floor=min(metrics['context_group']['directional_ratios_to_raw'])>=.98,
        old_anchor=candidate[1]>=anchor[1] and candidate[2]<=anchor[2],
        contextual_value=all(any([candidate[0]>values(metrics[a])[0],candidate[1]>values(metrics[a])[1],candidate[2]<values(metrics[a])[2]]) for a in ['global_group','shuffled_context']))
    base.write(ROOT/'summary.json',dict(metrics=metrics,controls=control_report,drift=drift,costs=dict(prior=fitcost,old_context_logz=lzcost,scoring=costs),
        total_seconds=time.monotonic()-begin,new_training_configs=0,new_parent_trainings=0,no_test=True,zero_offset_ranks_equal=True,
        gate=checks,decision='GO score comparability candidate only; update interface unfinished' if all(checks.values()) else 'NO-GO group calibration; no parameter repair',
        all_valid_development=True,inference_only_not_adaptive_training=True))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
