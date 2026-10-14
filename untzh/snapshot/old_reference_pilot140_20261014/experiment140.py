import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/risk_link136_20261014')
import experiment136 as risk
worker,base,core,cx=risk.worker,risk.base,risk.core,risk.cx
ROOT=Path(__file__).resolve().parent
SRC=Path('/data/lc/untzh/wn_transfer112_20261014')
N,TOTAL,D=24567,28660,200

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def split_old(old,pt):
    tick=time.monotonic()
    canonical=np.flatnonzero(old[:,1]%2==0)
    index={tuple(int(v) for v in a):i for i,a in enumerate(old)}
    assert len(index)==len(old) and len(canonical)*2==len(old)
    entity_degree=np.zeros(N,np.int64)
    relation_degree=np.zeros(11,np.int64)
    for h,r,t in old[canonical]:
        entity_degree[h]+=1
        if t!=h:entity_degree[t]+=1
        relation_degree[r//2]+=1
    before_degree=entity_degree.copy()
    before_relation=relation_degree.copy()
    keep=np.ones(len(old),bool)
    chosen=[]
    target=int(np.floor(.1*len(canonical)))
    for i in np.random.default_rng(1400017).permutation(canonical):
        h,r,t=old[i]
        if entity_degree[h]<=1 or entity_degree[t]<=1 or relation_degree[r//2]<=1:continue
        inv=index[(int(t),int(r)^1,int(h))]
        keep[i]=keep[inv]=False
        chosen.append(int(i))
        entity_degree[h]-=1
        if t!=h:entity_degree[t]-=1
        relation_degree[r//2]-=1
        if len(chosen)==target:break
    assert np.array_equal(before_degree>0,entity_degree>0)
    assert np.array_equal(before_relation>0,relation_degree>0)
    draw=keep[pt['pos']]
    cumulative=np.concatenate([[0],np.cumsum(draw,dtype=np.int64)])
    bounds=cumulative[pt['bounds']]
    assert np.all(np.diff(bounds)>0)
    filtered=dict(epochs=pt['epochs'],bounds=bounds,pos=pt['pos'][draw],nh=pt['nh'][draw],nt=pt['nt'][draw])
    assert all(keep[x] for x in filtered['pos'])
    fit_ids=np.sort(np.random.default_rng(1400018).choice(np.flatnonzero(keep),min(4096,int(keep.sum())),replace=False))
    held_ids=np.flatnonzero(~keep)
    np.savez(ROOT/'cache/split.npz',fit_mask=keep,fit_probe_ids=fit_ids,held_probe_ids=held_ids)
    np.savez(ROOT/'cache/parent-learning.npz',**filtered)
    report=dict(seed=1400017,target_canonical=target,held_canonical=len(chosen),held_directions=len(held_ids),fit_directions=int(keep.sum()),
                original_draws=len(pt['pos']),fit_draws=len(filtered['pos']),optimizer_steps=len(pt['epochs']),
                entities_preserved=True,relations_preserved=True,inverses_paired=True,seconds=time.monotonic()-tick,
                selection_bias_due_to_coverage=True,same_steps_not_same_draws=True)
    base.write(ROOT/'split.json',report)
    return filtered,fit_ids,held_ids,report

def parent_train(old,pt):
    torch.manual_seed(17)
    E=torch.nn.Parameter(torch.randn((N,D),device='cuda')*.12)
    R=torch.nn.Parameter(torch.randn((22,D),device='cuda')*.12)
    opt=torch.optim.Adam([E,R],lr=.007)
    torch.cuda.synchronize();tick=time.monotonic()
    for j in range(len(pt['epochs'])):
        opt.zero_grad(set_to_none=True)
        cx.learning(E,R,old,pt,j).backward()
        opt.step()
        assert torch.isfinite(E).all() and torch.isfinite(R).all()
        if j==len(pt['epochs'])-1 or pt['epochs'][j+1]!=pt['epochs'][j]:status('parent_train',epoch=int(pt['epochs'][j]))
    torch.cuda.synchronize()
    result=dict(E=E.detach().cpu(),R=R.detach().cpu())
    torch.save(result,ROOT/'cache/parent.pt')
    return result,dict(seconds=time.monotonic()-tick,steps=len(pt['epochs']),epochs=int(pt['epochs'][-1]))

def increment(parent,new,tape):
    initial=dict(E=torch.zeros((TOTAL,D),dtype=parent['E'].dtype),R=parent['R'])
    initial['E'][:N]=parent['E']
    torch.save(initial,ROOT/'cache/initial.pt')
    E,R,opt=worker.model(initial,N)
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    torch.cuda.synchronize();tick=time.monotonic()
    prefix=None
    for j in range(len(tape['epochs'])):
        opt.zero_grad(set_to_none=True)
        cx.learning(E,R,new,tape,j).backward()
        opt.step()
        assert torch.isfinite(E).all()
        if j==start-1:
            prefix=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()))
            torch.save(prefix,ROOT/'models/prefix10.pt')
    torch.cuda.synchronize()
    endpoint=dict(E=E.detach().cpu().double(),R=R.cpu().double())
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    torch.save(endpoint,ROOT/'models/epoch20.pt')
    return initial,prefix,endpoint,dict(seconds=time.monotonic()-tick,steps=len(tape['epochs']),frozen_equal=True)

@torch.no_grad()
def diagnose(name,ids,old,new,initial,prefix,endpoint,truths):
    tick=time.monotonic()
    facts=old[ids]
    cache,teacher_cost=cx.teacher_cache(initial,facts,truths,name)
    states={}
    stagecost={}
    for stage,m in [('expanded',initial),('prefix10',prefix),('epoch20',endpoint)]:
        states[stage],stagecost[stage]=cx.fast_ranks(m['E'].cuda().double(),m['R'].cuda().double(),facts,cache,truths)
    eligible=cache['teacher'][:,1]<=10
    q=cx.tail_query(initial['E'][facts[:,0]].double(),initial['R'][facts[:,1]].double())
    normgold=cache['gold']/q.norm(dim=1).clamp_min(1e-30).numpy()
    riskrows=np.zeros(TOTAL-N,bool)
    row_pairs=np.zeros(TOTAL-N,np.int64)
    Ep=prefix['E'][N:].cuda().double()
    for lo in range(0,len(facts),128):
        hi=min(lo+128,len(facts))
        competing=q[lo:hi].cuda()@Ep.T>torch.as_tensor(cache['gold'][lo:hi],device='cuda')[:,None]
        competing&=torch.as_tensor(eligible[lo:hi],device='cuda')[:,None]
        for i,(h,r,g) in enumerate(facts[lo:hi]):
            forbid=[c-N for c in truths[int(h),int(r)] if c>=N]
            if forbid:competing[i,forbid]=False
        riskrows|=competing.any(0).cpu().numpy()
        row_pairs+=competing.sum(0).cpu().numpy()
    degree,_=risk.degree(new,N,TOTAL)
    original=base.load('/data/lc/untzh/complex_transfer118_20261014/cache/fast-old_train.npz')['teacher'][ids,1]
    report=dict(probe_n=len(ids),eligible=int(eligible.sum()),eligible_fraction=float(eligible.mean()),
                full_parent_cached_probe_mrr=float((1/original).mean()),pilot_parent_probe_mrr=float((1/cache['teacher'][:,1]).mean()),
                eligible_gold=dict(negative=int((normgold[eligible]<0).sum()),negative_fraction=float((normgold[eligible]<0).mean()) if eligible.any() else None,
                                   quantiles={str(p):float(np.quantile(normgold[eligible],p)) for p in [.05,.5,.95]} if eligible.any() else {}),
                phases=worker.transitions(cache['teacher'][:,1],states['expanded'][:,1],states['prefix10'][:,1],states['epoch20'][:,1],eligible),
                prefix_row_pressure=dict(rows=int(riskrows.sum()),fraction=float(riskrows.mean()),
                    strata={key:dict(rows=int(mask.sum()),pressure_rows=int((mask&riskrows).sum()),pairs=int(row_pairs[mask].sum())) for key,mask in [('one',degree==1),('two_three',(degree>=2)&(degree<=3)),('four_plus',degree>=4)]}),
                teacher_cost=teacher_cost,stage_cost=stagecost,seconds=time.monotonic()-tick)
    np.savez(ROOT/'evaluation'/f'{name}.npz',ids=ids,teacher=cache['teacher'],gold=cache['gold'],normalized_gold=normgold,
             row_pressure=riskrows,row_pairs=row_pairs,**states)
    return report

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    cx.ROOT=ROOT
    for sub in ['cache','models','evaluation']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'cache/parent.pt').exists(), 'existing trained parent: do not silently rerun'
    status('prepare')
    z=base.load(SRC/'cache/train.npz')
    pt=base.load(SRC/'cache/parent-learning.npz')
    tape=base.load(SRC/'cache/learning.npz')
    pt,fit_ids,held_ids,split=split_old(z['old'],pt)
    parent,parent_cost=parent_train(z['old'],pt)
    del pt
    status('increment')
    initial,prefix,end,increment_cost=increment(parent,z['new'],tape)
    base.write(ROOT/'models-committed.json',dict(parent=True,ordinary=True,no_valid_read=True))
    truths=core.truthmap(z['old'],z['new'])
    reports={}
    for name,ids in [('fit_probe',fit_ids),('internal_reference',held_ids)]:
        status('diagnose',split=name)
        reports[name]=diagnose(name,ids,z['old'],z['new'],initial,prefix,end,truths)
        print('REFERENCE140',name,reports[name],flush=True)
    base.write(ROOT/'summary.json',dict(split=split,parent_training=parent_cost,increment_training=increment_cost,
        reports=reports,total_seconds=time.monotonic()-tick,new_training_configs=1,new_parent_trainings=1,
        no_valid=True,no_test=True,diagnostic_not_adaptive_method=True,decision_pending=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
