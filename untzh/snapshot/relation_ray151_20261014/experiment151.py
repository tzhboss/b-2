import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/relation_update150_20261014')
import experiment150 as prior
base,core=prior.base,prior.core
ROOT=Path(__file__).resolve().parent
N,TOTAL=prior.N,prior.TOTAL
ARMS=['ray_none','ray_answer_anchor','ray_full_replay']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def sync():torch.cuda.synchronize();return time.monotonic()

def model(prefix):
    E=torch.nn.Parameter(prefix['E'].cuda().clone())
    R=prefix['R'].cuda().clone()
    a=torch.nn.Parameter(torch.zeros(len(R),device='cuda',dtype=R.dtype))
    E.register_hook(lambda g:torch.cat([torch.zeros_like(g[:N]),g[N:]]))
    group=prefix['optimizer']['param_groups'][0]
    opt=torch.optim.Adam([E,a],lr=group['lr'],betas=group['betas'],eps=group['eps'],weight_decay=group['weight_decay'],amsgrad=group['amsgrad'])
    state=copy.deepcopy(prefix['optimizer']['state'][group['params'][0]])
    opt.state[E]={k:v.cuda() if torch.is_tensor(v) and v.ndim else v for k,v in state.items()}
    return E,R,a,opt

def learning(E,R,a,new,tape,j):
    lo,hi=tape['bounds'][j:j+2]
    p,nh,nt=[torch.as_tensor(x,device='cuda',dtype=torch.long) for x in (new[tape['pos'][lo:hi]],tape['nh'][lo:hi],tape['nt'][lo:hi])]
    h,r,t=p.unbind(1)
    scale=a[r].exp()
    ps=scale*(E[h]*R[r]*E[t]).sum(-1)
    ns=scale[:,None]*(E[nh]*R[r[:,None]]*E[nt]).sum(-1)
    return F.softplus(-ps).mean()+F.softplus(ns).mean()+.0005*(E[h].square().mean()+(R[r].square()*scale[:,None].square()).mean())

def anchor_loss(E,R,a,anchor,k):
    ix=anchor['selections'][k]
    p=anchor['facts'][ix]
    q=E[p[:,0]].double()*R[p[:,1]].double()
    scale=a[p[:,1]].exp().double()
    ss=scale[:,None]*(q@E[N:].double().T)
    legal=anchor['legal'][ix]
    candidate=(F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
    score=scale*(q*E[p[:,2]].double()).sum(1)
    return candidate+(score-anchor['teacher'][ix]).square().mean()

def replay_loss(E,R,a,replay,k):
    p=replay['facts'][k]
    h,r,t=p.unbind(1)
    scale=a[r].exp()
    tail=scale[:,None]*((E[h]*R[r])@E.T)
    head=scale[:,None]*((E[t]*R[r])@E.T)
    row=torch.arange(len(p),device='cuda')
    positive=F.softplus(-tail[row,t]).mean()
    values=[]
    for side,ss,den in [('tail',tail,replay['dt'][k]),('head',head,replay['dh'][k])]:
        cache=replay[side]
        lo,hi=cache['bounds'][[k*32,(k+1)*32]]
        mask=torch.ones_like(ss,dtype=torch.bool)
        mask[cache['rows'][lo:hi]-k*32,cache['cols'][lo:hi]]=False
        values.append((F.softplus(ss).masked_fill(~mask,0).sum(1)/den).mean())
    return positive+.5*(values[0]+values[1])+.0005*(E[h].square().mean()+(R[r].square()*scale[:,None].square()).mean())

def train(arm,initial,prefix,new,tape,anchor,replay):
    E,R,a,opt=model(prefix)
    se,sr,sa,so=model(prefix)
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    records=[]
    costs={k:0. for k in ['learning','both_protection_gradients','update','ordinary_shadow','recording']}
    begin=sync()
    for j in range(start,len(tape['epochs'])):
        k=j-start
        tick=sync()
        gl=torch.autograd.grad(learning(E,R,a,new,tape,j),[E,a])
        costs['learning']+=sync()-tick
        tick=sync()
        ga=torch.autograd.grad(anchor_loss(E,R,a,anchor,k),[E,a])
        gr=torch.autograd.grad(replay_loss(E,R,a,replay,k),[E,a])
        costs['both_protection_gradients']+=sync()-tick
        gp=ga if arm=='ray_answer_anchor' else gr if arm=='ray_full_replay' else [torch.zeros_like(g) for g in gl]
        bn,pn=float(prior.norm(gl)),float(prior.norm(gp))
        alpha=.1*bn/pn if pn else 0.
        tick=sync()
        opt.zero_grad(set_to_none=True)
        E.grad=gl[0]+alpha*gp[0]
        a.grad=gl[1]+alpha*gp[1]
        opt.step()
        costs['update']+=sync()-tick
        tick=sync()
        so.zero_grad(set_to_none=True)
        learning(se,sr,sa,new,tape,j).backward()
        so.step()
        costs['ordinary_shadow']+=sync()-tick
        tick=sync()
        scale=a.detach().exp()
        assert torch.isfinite(E).all() and torch.isfinite(scale).all() and (scale>0).all()
        records.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,scale_min=float(scale.min()),scale_mean=float(scale.mean()),scale_max=float(scale.max()),
            deltaE=float((E.detach()[N:]-se.detach()[N:]).double().norm()),delta_log_scale=float((a.detach()-sa.detach()).double().norm())))
        costs['recording']+=sync()-tick
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    dest=ROOT/'models'/arm
    dest.mkdir(parents=True,exist_ok=True)
    payload=dict(E=E.detach().cpu().double(),R=R.cpu().double(),relation_scale=a.detach().exp().cpu().double(),
                 log_relation_scale=a.detach().cpu(),score_definition='relation_scale[r] * base DistMult score; positive scale cancels in ranking')
    torch.save(payload,dest/'epoch-20.pt')
    torch.save(dict(E=se.detach().cpu().double(),R=sr.cpu().double(),relation_scale=sa.detach().exp().cpu().double()),dest/'ordinary-shadow.pt')
    base.write(dest/'steps.json',records)
    report=dict(seconds=sync()-begin,phase_costs=costs,steps=len(records),nonzero_steps=sum(r['ratio']>0 for r in records),
        mean_ratio=float(np.mean([r['ratio'] for r in records])),scale_quantiles={str(q):float(torch.quantile(a.detach().exp().double(),q)) for q in [0.,.1,.5,.9,1.]},
        newE_step=int(opt.state[E]['step']),scale_step=int(opt.state[a]['step']),oldE_frozen=True,relation_direction_frozen=True,
        positive_scale_learned=True,shadow_compute_included=True)
    base.write(dest/'training.json',report)
    return report

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
    # Reuse150 cached full replay indices; only small anchor tensor preparation repeated.
    prior.ROOT=ROOT
    anchor,replay,prep=prior.prepare(initial,old,truths)
    training={}
    for arm in ARMS:
        status('train',arm=arm)
        training[arm]=train(arm,initial,prefix,new,tape,anchor,replay)
        print('TRAIN151',arm,training[arm],flush=True)
    ordinary=torch.load(ROOT/'models/ray_none/epoch-20.pt',map_location='cpu',weights_only=True)
    for arm in ARMS:
        shadow=torch.load(ROOT/'models'/arm/'ordinary-shadow.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(shadow[k],ordinary[k]) for k in ['E','R','relation_scale'])
    base.write(ROOT/'committed-models.json',dict(new_training_configs=3,all_models_before_valid=True,all_shadows_equal=True,
             positive_relation_scaling_only=True,scale_cancelled_rank_evaluation=True))
    full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
    fast=prior.previous.controls.pair.fastmodule.FastOld(17,truths)
    queries=dict(fast.queries,new_valid=full['new_valid'])
    metrics={};drift={};ec={}
    (ROOT/'evaluation').mkdir(exist_ok=True)
    for arm in ARMS:
        state=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        E,R=state['E'].cuda(),state['R'].cuda()
        scale=state['relation_scale'].cuda()
        # Canonical score is scale[r] times the entire unscaled score row.
        sample=full['new_valid'][:64]
        q=E[sample[:,0]]*R[sample[:,1]]
        raw=q@E.T
        scaled=raw*scale[sample[:,1],None]
        idx=torch.as_tensor(sample[:,2],device='cuda');row=torch.arange(len(sample),device='cuda')
        assert torch.equal((raw>raw[row,idx,None]).sum(1),(scaled>scaled[row,idx,None]).sum(1))
        ranks={};costs={}
        for split in ['old_train','old_valid']:
            status('evaluate',arm=arm,split=split)
            ranks[split],costs[split]=fast.ranks(E,R,split)
        st=time.monotonic()
        status('evaluate',arm=arm,split='new_valid')
        ranks['new_valid']=base.ranks(E,R,full['new_valid'],truths)
        costs['new_valid']=dict(seconds=time.monotonic()-st)
        np.savez(ROOT/'evaluation'/f'{arm}.npz',**ranks)
        m={s:base.metrics(ranks[s],facts) for s,facts in queries.items()}
        for s in fast.teacher:m[s]['drops']=core.drops(ranks[s],fast.teacher[s])
        metrics[arm]=m;ec[arm]=costs;drift[arm]={s:prior.previous.drift(ranks[s],fast.teacher[s]) for s in fast.teacher}
        print('EVALUATED151',arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
    import json
    reference=json.loads(Path('/data/lc/untzh/relation_update150_20261014/summary.json').read_text())['frozen_permission_references']
    def vals(m):return (m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'])
    plain=vals(metrics['ray_none']);fplain=vals(reference['none']);fanchor=vals(reference['anchor'])
    checks={}
    for arm in ARMS:
        m=metrics[arm];v=vals(m)
        dr=[m['new_valid'][d]['filtered_train_truths']['mrr']/metrics['ray_none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        m['directional_ratios_to_same_permission_none']=dr
        checks[arm]=dict(same_permission_learning_floor=v[0]>=plain[0]*.98 and min(dr)>=.98,
            same_permission_old_not_worse=v[1]>=plain[1] and v[2]<=plain[2],different_permission_learning_floor=v[0]>=fplain[0]*.98,
            different_permission_old_reference=v[1]>=fanchor[1] and v[2]<=fanchor[2],
            at_least_one_reference_gain=v[0]>fanchor[0] or v[1]>fanchor[1])
    go=any(all(checks[a].values()) for a in ARMS[1:])
    base.write(ROOT/'summary.json',dict(metrics=metrics,drift=drift,costs=dict(preparation=prep,training=training,evaluation=ec),
        checks=checks,frozen_permission_references=reference,total_seconds=time.monotonic()-begin,new_training_configs=3,new_parent_trainings=0,
        old_candidate_ranks_invariant_by_positive_scaling=True,all_valid_development=True,no_test=True,
        decision='GO rank invariant freedom mechanism only; no adaptive benefit proven' if go else 'NO-GO positive relation scale condition; no parameter repair'))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
