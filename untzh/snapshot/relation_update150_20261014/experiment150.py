import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/forecast_protect146_20261014')
import experiment146 as previous
base,core=previous.base,previous.core
ROOT=Path(__file__).resolve().parent
N,TOTAL=previous.N,previous.TOTAL
ARMS=['relation_none','relation_answer_anchor','relation_full_replay']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def sync():torch.cuda.synchronize();return time.monotonic()

def model(initial,prefix):
    E=torch.nn.Parameter(prefix['E'].cuda().clone())
    R=torch.nn.Parameter(prefix['R'].cuda().clone())
    E.register_hook(lambda g:torch.cat([torch.zeros_like(g[:N]),g[N:]]))
    group=prefix['optimizer']['param_groups'][0]
    opt=torch.optim.Adam([E,R],lr=group['lr'],betas=group['betas'],eps=group['eps'],weight_decay=group['weight_decay'],amsgrad=group['amsgrad'])
    state=copy.deepcopy(prefix['optimizer']['state'][group['params'][0]])
    opt.state[E]={k:v.cuda() if torch.is_tensor(v) and v.ndim else v for k,v in state.items()}
    assert R not in opt.state
    return E,R,opt

def norm(grads):return torch.sqrt(sum(g.double().square().sum() for g in grads))

def sparse_indices(facts,truths,include_gold=True,heads=False):
    row=[];col=[]
    bounds=[0]
    for i,(h,r,t) in enumerate(facts):
        candidates=truths[int(t),int(r)^1] if heads else truths[int(h),int(r)]
        gold=int(h if heads else t)
        for v in candidates:
            if (include_gold or v!=gold) and v<TOTAL:row.append(i);col.append(v)
        bounds.append(len(row))
    return dict(rows=np.asarray(row,dtype=np.int64),cols=np.asarray(col,dtype=np.int64),bounds=np.asarray(bounds,dtype=np.int64))

def prepare(initial,old,truths):
    tick=sync()
    src=Path('/data/lc/untzh/increment87_20261014')
    c=base.load(src/'cache/cohort.npz')
    legal=np.ones((len(c['ids']),TOTAL-N),bool)
    for i,(h,r,g) in enumerate(c['queries']):
        forbidden=[v-N for v in truths[int(h),int(r)] if v>=N]
        if forbidden:legal[i,forbidden]=False
    facts=torch.as_tensor(c['queries'],device='cuda')
    teacher=(initial['E'][c['queries'][:,0]].double()*initial['R'][c['queries'][:,1]].double()*initial['E'][c['queries'][:,2]].double()).sum(1)
    anchor=dict(facts=facts,teacher=teacher.cuda(),legal=torch.as_tensor(legal,device='cuda'),
                selections=base.load(src/'cache/protection.npz')['selections'][:220])
    rp=base.load('/data/lc/untzh/full_replay105_20261014/seed17/cache/full-replay.npz')['positions']
    assert len(rp)==220*32
    p=old[rp]
    assert all(int(h) in truths[int(t),int(r)^1] for h,r,t in p)
    rt=sparse_indices(p,truths)
    rh=sparse_indices(p,truths,heads=True)
    dt=TOTAL-np.diff(rt['bounds']);dh=TOTAL-np.diff(rh['bounds'])
    replay=dict(facts=torch.as_tensor(p.reshape(220,32,3),device='cuda'),tail=rt,head=rh,
                dt=torch.as_tensor(dt.reshape(220,32),device='cuda'),dh=torch.as_tensor(dh.reshape(220,32),device='cuda'))
    (ROOT/'cache').mkdir(exist_ok=True)
    np.savez(ROOT/'cache/fixed-replay.npz',positions=rp,facts=p,tail_rows=rt['rows'],tail_cols=rt['cols'],tail_bounds=rt['bounds'],
             head_rows=rh['rows'],head_cols=rh['cols'],head_bounds=rh['bounds'])
    return anchor,replay,dict(seconds=sync()-tick,old_facts_reused=True,replay_draws=len(p),old_terms_retained=True,
                             no_valid_input=True,sparse_filters_cached=True)

def anchor_loss(E,R,anchor,k):
    ix=anchor['selections'][k]
    p=anchor['facts'][ix]
    q=E[p[:,0]].double()*R[p[:,1]].double()
    ss=q@E[N:].double().T
    legal=anchor['legal'][ix]
    candidate=(F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
    score=(q*E[p[:,2]].double()).sum(1)
    return candidate+(score-anchor['teacher'][ix]).square().mean()

def replay_loss(E,R,replay,k):
    p=replay['facts'][k]
    h,r,t=p.unbind(1)
    tail=(E[h]*R[r])@E.T
    head=(E[t]*R[r])@E.T
    row=torch.arange(len(p),device='cuda')
    positive=F.softplus(-tail[row,t]).mean()
    values=[]
    for side,ss,den in [('tail',tail,replay['dt'][k]),('head',head,replay['dh'][k])]:
        cache=replay[side]
        lo,hi=cache['bounds'][k*32:(k+1)*32+1][[0,-1]]
        mask=torch.ones_like(ss,dtype=torch.bool)
        mask[cache['rows'][lo:hi]-k*32,cache['cols'][lo:hi]]=False
        values.append((F.softplus(ss).masked_fill(~mask,0).sum(1)/den).mean())
    return positive+.5*(values[0]+values[1])+.0005*(E[h].square().mean()+R[r].square().mean())

def train(arm,initial,prefix,new,tape,anchor,replay):
    E,R,opt=model(initial,prefix)
    se,sr,so=model(initial,prefix)
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    begin=sync()
    records=[]
    costs={k:0. for k in ['learning','both_protection_gradients','update','ordinary_shadow','recording']}
    for j in range(start,len(tape['epochs'])):
        k=j-start
        tick=sync()
        loss=base.learning(E,R,new,tape,j)
        gl=torch.autograd.grad(loss,[E,R])
        costs['learning']+=sync()-tick
        tick=sync()
        pa=anchor_loss(E,R,anchor,k)
        ga=torch.autograd.grad(pa,[E,R])
        pr=replay_loss(E,R,replay,k)
        gr=torch.autograd.grad(pr,[E,R])
        costs['both_protection_gradients']+=sync()-tick
        gp=ga if arm=='relation_answer_anchor' else gr if arm=='relation_full_replay' else [torch.zeros_like(x) for x in gl]
        bn,pn=float(norm(gl)),float(norm(gp))
        alpha=.1*bn/pn if pn else 0.
        tick=sync()
        opt.zero_grad(set_to_none=True)
        E.grad=gl[0]+alpha*gp[0]
        R.grad=gl[1]+alpha*gp[1]
        opt.step()
        costs['update']+=sync()-tick
        tick=sync()
        so.zero_grad(set_to_none=True)
        base.learning(se,sr,new,tape,j).backward()
        so.step()
        costs['ordinary_shadow']+=sync()-tick
        tick=sync()
        assert torch.isfinite(E).all() and torch.isfinite(R).all()
        records.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,learning_newE_norm=float(gl[0][N:].double().norm()),
            learning_R_norm=float(gl[1].double().norm()),protect_R_norm=float(gp[1].double().norm()),
            teacher_R_distance=float((R.detach()-initial['R'].cuda()).double().norm()),
            deltaE=float((E.detach()[N:]-se.detach()[N:]).double().norm()),deltaR=float((R.detach()-sr.detach()).double().norm())))
        costs['recording']+=sync()-tick
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(se[:N].cpu(),initial['E'][:N])
    dest=ROOT/'models'/arm
    dest.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.detach().cpu().double()),dest/'epoch-20.pt')
    torch.save(dict(E=se.detach().cpu().double(),R=sr.detach().cpu().double()),dest/'ordinary-shadow.pt')
    base.write(dest/'steps.json',records)
    report=dict(seconds=sync()-begin,phase_costs=costs,steps=len(records),nonzero_steps=sum(r['ratio']>0 for r in records),
        mean_ratio=float(np.mean([r['ratio'] for r in records])),final_R_teacher_distance=records[-1]['teacher_R_distance'],
        newE_step=int(opt.state[E]['step']),R_step=int(opt.state[R]['step']),oldE_frozen=True,R_trainable=True,
        same_available_parameters=True,shadow_compute_included=True)
    base.write(dest/'training.json',report)
    return report

@torch.no_grad()
def ranks(E,R,facts,filtercache):
    begin=sync()
    parts=[]
    for lo in range(0,len(facts),128):
        b=facts[lo:lo+128]
        q=torch.as_tensor(b,device='cuda')
        scores=(E[q[:,0]]*R[q[:,1]])@E.T
        gold=scores[torch.arange(len(b),device='cuda'),q[:,2]].clone()
        raw=1+(scores>gold[:,None]).sum(1)
        lower,upper=filtercache['bounds'][[lo,lo+len(b)]]
        if upper>lower:scores[filtercache['rows'][lower:upper]-lo,filtercache['cols'][lower:upper]]=-torch.inf
        filtered=1+(scores>gold[:,None]).sum(1)
        parts.append(torch.stack([raw,filtered],1).cpu().numpy())
    return np.concatenate(parts),dict(seconds=sync()-begin,queries=len(facts),all_candidates=TOTAL,old_R_change_fully_scored=True)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    base.N=core.N=N;base.TOTAL=core.TOTAL=TOTAL
    status('prepare')
    z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    old,new=z['old'],z['new']
    truths=core.truthmap(old,np.concatenate([new,z['extra']]))
    path=Path('/data/lc/untzh/train_init121_20261014/seed17/models/zero')
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(path/'prefix10.pt',map_location='cpu',weights_only=True)
    tape=base.load('/data/lc/untzh/increment87_20261014/cache/learning.npz')
    anchor,replay,prep=prepare(initial,old,truths)
    training={}
    for arm in ARMS:
        status('train',arm=arm)
        training[arm]=train(arm,initial,prefix,new,tape,anchor,replay)
        print('TRAIN150',arm,training[arm],flush=True)
    ordinary=torch.load(ROOT/'models/relation_none/epoch-20.pt',map_location='cpu',weights_only=True)
    for arm in ARMS:
        shadow=torch.load(ROOT/'models'/arm/'ordinary-shadow.pt',map_location='cpu',weights_only=True)
        assert torch.equal(shadow['E'],ordinary['E']) and torch.equal(shadow['R'],ordinary['R'])
    base.write(ROOT/'committed-models.json',dict(new_training_configs=3,all_endpoints_before_valid=True,all_ordinary_shadows_equal=True,
                 same_available_parameters=True,E_moment_history_reused=True,R_fresh_step=True))
    full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
    queries=dict(old_train=old,old_valid=full['old_valid'],new_valid=full['new_valid'])
    teacher=dict(old_train=base.load('/data/lc/untzh/increment87_20261014/evaluation/teacher.npz')['old_train'],
                 old_valid=base.load('/data/lc/untzh/full_valid102_20261014/seed17/teacher.npz')['old_valid'])
    filters={};filtercost={}
    baseline=base.load(path.parent.parent/'evaluation/zero.npz')
    for split,facts in queries.items():
        status('filter_cache',split=split)
        tick=time.monotonic()
        filters[split]=sparse_indices(facts,truths,include_gold=False)
        np.savez(ROOT/'cache'/f'{split}-filter.npz',**filters[split])
        filtercost[split]=dict(seconds=time.monotonic()-tick,known_filter_pairs=len(filters[split]['rows']),cached_once=True)
        sample,_=ranks(initial['E'].cuda().double(),initial['R'].cuda().double(),facts[:64],sparse_indices(facts[:64],truths,include_gold=False))
        reference=base.ranks(initial['E'].cuda().double(),initial['R'].cuda().double(),facts[:64],truths)
        assert np.array_equal(sample,reference)
    metrics={};ec={};drift={}
    (ROOT/'evaluation').mkdir(exist_ok=True)
    for arm in ARMS:
        modelstate=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        E,R=modelstate['E'].cuda(),modelstate['R'].cuda()
        states={};costs={}
        for split,facts in queries.items():
            status('evaluate',arm=arm,split=split)
            states[split],costs[split]=ranks(E,R,facts,filters[split])
        np.savez(ROOT/'evaluation'/f'{arm}.npz',**states)
        m={s:base.metrics(states[s],facts) for s,facts in queries.items()}
        for s in teacher:m[s]['drops']=core.drops(states[s],teacher[s])
        metrics[arm]=m;ec[arm]=costs
        drift[arm]={s:previous.drift(states[s],teacher[s]) for s in teacher}
        print('EVALUATED150',arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
    frozen={}
    for name,rankpath in [('none',path.parent.parent/'evaluation/zero.npz'),
        ('anchor',Path('/data/lc/untzh/smallinit_protect126_20261014/seed17/zero/evaluation/anchor_null.npz')),
        ('replay',Path('/data/lc/untzh/forecast_protect146_20261014/seed17/evaluation/fixed_replay.npz'))]:
        states=base.load(rankpath)
        m={s:base.metrics(states[s],facts) for s,facts in queries.items()}
        for s in teacher:m[s]['drops']=core.drops(states[s],teacher[s])
        frozen[name]=m
    def vals(m):return (m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'])
    checks={}
    plain=vals(metrics['relation_none']);fplain=vals(frozen['none']);fanchor=vals(frozen['anchor'])
    for arm in ARMS:
        m=metrics[arm];v=vals(m)
        dr=[m['new_valid'][d]['filtered_train_truths']['mrr']/metrics['relation_none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        m['directional_ratios_to_same_permission_none']=dr
        checks[arm]=dict(same_permission_learning_floor=v[0]>=plain[0]*.98 and min(dr)>=.98,
            same_permission_old_not_worse=v[1]>=plain[1] and v[2]<=plain[2],
            different_permission_learning_floor=v[0]>=fplain[0]*.98,
            different_permission_old_reference=v[1]>=fanchor[1] and v[2]<=fanchor[2],
            at_least_one_reference_gain=v[0]>fanchor[0] or v[1]>fanchor[1])
    go=any(all(checks[a].values()) for a in ARMS[1:])
    base.write(ROOT/'summary.json',dict(metrics=metrics,frozen_permission_references=frozen,drift=drift,
        costs=dict(preparation=prep,training=training,filter_preparation=filtercost,evaluation=ec),checks=checks,
        total_seconds=time.monotonic()-begin,new_training_configs=3,new_parent_trainings=0,all_valid_development=True,no_test=True,
        decision='GO relation freedom mechanism only; adaptive benefit unproven' if go else 'NO-GO relation release condition; no learning-rate or penalty repair'))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
