import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
import joblib
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
sys.path.insert(0,'/data/lc/untzh/blend_confirm80_20261014')
import experiment80 as core
base=core.base
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/wn_transfer112_20261014')
N,TOTAL=24567,28660

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

@torch.no_grad()
def features(initial,prefix,queries,teacher,gold_cache,truths,train):
    tick=time.monotonic();count=len(queries);rng=np.random.default_rng(1130017)
    candidates=rng.integers(N,TOTAL,size=(count,128));oldsample=rng.integers(N,size=(count,64));rows=[];cols=[]
    for i,(h,r,g) in enumerate(queries):
        known=list(truths[int(h),int(r)]);bad=np.isin(candidates[i],known)
        while bad.any():candidates[i,bad]=rng.integers(N,TOTAL,size=int(bad.sum()));bad=np.isin(candidates[i],known)
        for c in known:
            if c>=N:rows.append(i);cols.append(c-N)
    e=initial['E'].double().cuda();rel=initial['R'].double().cuda();newE=prefix['E'].double().cuda();q=e[queries[:,0]]*rel[queries[:,1]]
    hc=np.bincount(train[:,0],minlength=TOTAL);gc=np.bincount(train[:,2],minlength=TOTAL);rc=np.bincount(train[:,1],minlength=len(rel))
    X=[];ranks=[];rows=np.array(rows,dtype=np.int64);cols=np.array(cols,dtype=np.int64)
    for lo in range(0,count,128):
        hi=min(lo+128,count);qb=q[lo:hi];b=queries[lo:hi];gold=torch.as_tensor(gold_cache[lo:hi],device='cuda')
        # Scale is a fixed sample of old scores, identical feature definition to058.
        oldids=oldsample[lo:hi];tau=(qb[:,None]*e[oldids]).sum(-1).std(-1).clamp_min(1e-8)
        ids=candidates[lo:hi];z=((qb[:,None]*newE[ids]).sum(-1)-gold[:,None])/tau[:,None]
        ss=qb@newE[N:].T;above=ss>gold[:,None];i=np.searchsorted(rows,lo);j=np.searchsorted(rows,hi)
        if j>i:above[rows[i:j]-lo,cols[i:j]]=False
        ranks.append(teacher[lo:hi]+above.sum(-1).cpu().numpy())
        known=np.array([sum(c>=N for c in truths[int(h),int(r)]) for h,r,g in b])
        fixed=np.column_stack([teacher[lo:hi],11-teacher[lo:hi],(gold/tau).cpu().numpy(),qb.norm(dim=-1).cpu().numpy(),np.log1p(hc[b[:,0]]),np.log1p(rc[b[:,1]]),np.log1p(gc[b[:,2]]),known])
        current=torch.stack([z[:,:32].max(-1).values,z.max(-1).values,z.topk(4,dim=-1).values.mean(-1),z.mean(-1),z.std(-1),(z>0).double().mean(-1),F.softplus(z).mean(-1)],1).float().cpu().numpy()
        block=np.column_stack([fixed,current,np.zeros((len(b),5)),np.full(len(b),.5)]).astype(np.float32)
        assert block.shape==(len(b),21) and np.isfinite(block).all();X.append(block)
    payload=dict(X=np.concatenate(X),prefix_rank=np.concatenate(ranks),candidates=candidates,oldsample=oldsample,queries=queries)
    return payload,dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in payload.values()))

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();status('prepare');(ROOT/'cache').mkdir()
    initial=torch.load(SOURCE/'cache/initial.pt',map_location='cpu',weights_only=True);prefix=torch.load(SOURCE/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    train=base.load(SOURCE/'cache/train.npz');alltrain=np.concatenate([train['old'],train['new']]);truths=core.truthmap(train['old'],train['new'])
    queries=base.load(SOURCE/'evaluation/queries.npz');teachers=base.load(SOURCE/'evaluation/teacher.npz');cohort=base.load(SOURCE/'cache/cohort.npz')
    learner=joblib.load('/data/lc/untzh/risk58_20261013/cache/tree_current.joblib');reports={}
    for group in ['TRAIN_cohort','VALID_reference']:
        status('features',group=group)
        ids=cohort['ids'] if group=='TRAIN_cohort' else np.flatnonzero(teachers['old_valid'][:,1]<=10)
        split='old_train' if group=='TRAIN_cohort' else 'old_valid';q=queries[split][ids];teacher=teachers[split][ids,1]
        gold=base.load(SOURCE/'cache'/f'fast-{split}.npz')['gold'][ids]
        payload,cost=features(initial,prefix,q,teacher,gold,truths,alltrain);X=payload['X'];np.savez(ROOT/'cache'/f'{group}.npz',ids=ids,**payload)
        scores={k:v for k,v in core.ex.rule_scores(X).items() if k in ['boundary','pressure32','pressure128']}
        scores['random']=np.random.default_rng(1130200).random(len(X));st=time.monotonic();scores['transfer_tree']=learner.predict_proba(X)[:,1];prediction_seconds=time.monotonic()-st
        # First endpoint labels are read after all predictions for this group exist.
        final=base.load(SOURCE/'evaluation/none.npz')[split][ids,1]
        safe=payload['prefix_rank']<=10;events=safe&(final>10);comparison={}
        for name,score in scores.items():
            ap=float(average_precision_score(final[safe]>10,score[safe])) if events.sum() and safe.sum()>events.sum() else None
            budgets={}
            for fraction in [.05,.1,.2]:
                k=int(np.ceil(len(score)*fraction));chosen=np.argsort(-score,kind='stable')[:k];hits=int(events[chosen].sum())
                budgets[str(fraction)]=dict(selected=k,hits=hits,events=int(events.sum()),recall=hits/int(events.sum()) if events.sum() else None,precision=hits/k,already_dropped_selected=int((~safe[chosen]).sum()))
            comparison[name]=dict(safe_ap=ap,budgets=budgets)
        reports[group]=dict(n=len(q),safe_at_prefix=int(safe.sum()),already_dropped=int((~safe).sum()),final_drops=int((final>10).sum()),new_events=int(events.sum()),features=cost,transfer_prediction_seconds=prediction_seconds,methods=comparison)
        base.write(ROOT/'summary-running.json',dict(groups=reports));print('RISK113',group,reports[group]['new_events'],{n:v['budgets']['0.05']['hits'] for n,v in comparison.items()},flush=True)
    base.write(ROOT/'summary.json',dict(groups=reports,total_seconds=time.monotonic()-tick,no_training=True,historical_learner_cost_not_free=True));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
