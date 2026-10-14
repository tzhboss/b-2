import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/teacher_boundary134_20261014')
import experiment134 as previous
worker,base,core,cx=previous.worker,previous.base,previous.core,previous.cx
ROOT=Path(__file__).resolve().parent
ARMS=['none','fixed_anchor','fixed_replay','random_matched','risk_adaptive','degree_only']

def status(stage,**kw):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def sync():
    torch.cuda.synchronize()
    return time.monotonic()

def replay_assets(parent):
    path=Path('/data/lc/untzh/full_replay105_20261014/seed17/cache/full-replay.npz') if parent==17 else Path('/data/lc/untzh/complex_transfer118_20261014/cache/full-replay.npz')
    z=base.load(path)
    if parent==17:
        translate=dict(qt='qtail',qh='qhead',mt='mask_tail',mh='mask_head',dt='denominator_tail',dh='denominator_head')
        z={k:z[v] for k,v in translate.items()}
    return {k:torch.as_tensor(z[k],device='cuda') for k in ['qt','qh','mt','mh','dt','dh']}

def actions(gate,arm):
    if arm=='none':return np.zeros_like(gate['action'])
    if arm=='fixed_anchor':return np.ones_like(gate['action'])
    if arm=='fixed_replay':return np.full_like(gate['action'],2)
    if arm=='risk_adaptive':return gate['action'].copy()
    if arm=='random_matched':return np.random.default_rng(1370017).permutation(gate['action'])
    return np.where(gate['query_support']&(gate['degree']<=3),1,2).astype(np.int8)

def train(parent,arm,initial,prefix,ordinary,n,new,tape,anchor,replay,space,action,out):
    E,R,opt=worker.model(initial,n,prefix)
    se,sr,so=worker.model(initial,n,prefix)
    learning=cx.learning if parent=='WN_ComplEx' else base.learning
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    a=torch.as_tensor(action,device='cuda')
    total=sync()
    costs={k:0. for k in ['learning','anchor_gradient','replay_gradient','action_update','ordinary_shadow','projection','recording']}
    records=[]
    for j in range(start,len(tape['epochs'])):
        tick=sync()
        gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        costs['learning']+=sync()-tick
        k=j-start
        chosen=anchor['selections'][k]
        tick=sync()
        ss=anchor['q'][chosen]@E[n:].double().T
        legal=anchor['legal'][chosen]
        pa=(F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
        ga=torch.autograd.grad(pa,E)[0]
        costs['anchor_gradient']+=sync()-tick
        tick=sync()
        tail=F.softplus(replay['qt'][k]@E[n:].double().T).masked_fill(~replay['mt'][k],0).sum(1)/replay['dt'][k]
        head=F.softplus(replay['qh'][k]@E[n:].double().T).masked_fill(~replay['mh'][k],0).sum(1)/replay['dh'][k]
        pr=.5*(tail+head).mean()
        gr=torch.autograd.grad(pr,E)[0]
        costs['replay_gradient']+=sync()-tick
        tick=sync()
        gp=torch.zeros_like(gl)
        gp[n:]=torch.where((a==1)[:,None],ga[n:],torch.where((a==2)[:,None],gr[n:],torch.zeros_like(gr[n:])))
        bn,pn=float(gl[n:].double().norm()),float(gp[n:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True)
        E.grad=gl+alpha*gp
        opt.step()
        costs['action_update']+=sync()-tick
        tick=sync()
        so.zero_grad(set_to_none=True)
        learning(se,sr,new,tape,j).backward()
        so.step()
        costs['ordinary_shadow']+=sync()-tick
        tick=sync()
        ref=se.detach()[n:]
        raw=(E.detach()[n:]-ref).double()
        projected=cx.source.space_module.project(raw,space)
        # Leave non-Anchor host proposals untouched, avoiding a FP32 subtract/add round trip.
        selected_anchor=a==1
        with torch.no_grad():E[n:][selected_anchor]=ref[selected_anchor]+projected[selected_anchor].to(E.dtype)
        costs['projection']+=sync()-tick
        tick=sync()
        delta=(E.detach()[n:]-ref).double()
        changes=(space['vectors']*delta[space['rows']]).sum(1)
        assert bool(torch.isfinite(E).all())
        records.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,gp_norm=pn,delta_norm=float(delta.norm()),cross_max=float(changes.abs().max())))
        costs['recording']+=sync()-tick
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    directory=out/'models'/arm
    directory.mkdir(parents=True,exist_ok=True)
    endpoint=dict(E=E.detach().cpu().double(),R=R.cpu().double())
    torch.save(endpoint,directory/'epoch-20.pt')
    base.write(directory/'steps.json',records)
    report=dict(seconds=sync()-total,phase_costs=costs,steps=len(records),nonzero_steps=sum(r['gp_norm']>0 for r in records),
        mean_ratio=float(np.mean([r['ratio'] for r in records])),actions={str(code):int((action==code).sum()) for code in [0,1,2]},
        row_protection_fraction=float((action>0).mean()),anchor_fraction=float((action==1).mean()),replay_fraction=float((action==2).mean()),
        all_arms_both_protection_gradients_computed=True,shadow_compute_included=True,shadow_equal=True,frozen_equal=True)
    base.write(directory/'training.json',report)
    return report

def drift(state,teacher):
    result={}
    for mode,col in [('raw',0),('filtered_train_truths',1)]:
        b,a=teacher[:,col],state[:,col]
        eligible=b<=10
        result[mode]=dict(n=len(a),worsened=int((a>b).sum()),improved=int((a<b).sum()),mean_rank_delta=float((a-b).mean()),
            mean_log_rank_ratio=float(np.log(a/b).mean()),mean_rr_delta=float((1/a-1/b).mean()),
            eligible=int(eligible.sum()),eligible_worsened=int(((a>b)&eligible).sum()))
    return result

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    bundles=[]
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        prepstart=sync()
        n,total,src,new,tape,truths,anchor,space=worker.training_assets(parent)
        ip,pp,op,rp,ap=previous.paths(parent)
        initial=torch.load(ip,map_location='cpu',weights_only=True)
        prefix=torch.load(pp,map_location='cpu',weights_only=True)
        ordinary=torch.load(op,map_location='cpu',weights_only=True)
        replay=replay_assets(parent)
        gate=base.load(f'/data/lc/untzh/risk_link136_20261014/{parent}/cache/row-gate.npz')
        out=ROOT/str(parent)
        out.mkdir(parents=True,exist_ok=True)
        plans={arm:actions(gate,arm) for arm in ARMS}
        np.savez(out/'fixed-actions.npz',**plans)
        prep=dict(seconds=sync()-prepstart,risk_diagnostic136_not_free=True,no_resampling=True)
        training={}
        for arm in ARMS:
            status('train',parent=str(parent),arm=arm)
            training[arm]=train(parent,arm,initial,prefix,ordinary,n,new,tape,anchor,replay,space,plans[arm],out)
            print('TRAIN137',parent,arm,training[arm],flush=True)
        bundles.append((parent,n,total,src,truths,out,prep,training,rp,ap))
    base.write(ROOT/'committed-models.json',dict(new_training_configs=12,all_endpoints_before_valid=True,equal_compute_envelope=True))
    reports={}
    for parent,n,total,src,truths,out,prep,training,rp,ap in bundles:
        base.N=core.N=n
        base.TOTAL=core.TOTAL=total
        split=base.load(f'/data/lc/untzh/admission_action128_20261014/cache/{parent}-split.npz')
        if parent==17:
            worker.fastmodule.ROOT=Path('/data/lc/untzh/fast_old_eval107_20261014')
            fast=worker.fastmodule.FastOld(17,truths)
            queries=dict(fast.queries,new_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['new_valid'])
            teacher=fast.teacher
            oldrank=lambda E,R,s:fast.ranks(E,R,s)
            newrank=lambda E,R:base.ranks(E,R,queries['new_valid'],truths)
        else:
            queries=base.load(src/'evaluation/queries.npz')
            cache={s:base.load(src/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']}
            teacher={s:cache[s]['teacher'] for s in cache}
            oldrank=lambda E,R,s:cx.fast_ranks(E,R,queries[s],cache[s],truths)
            newrank=lambda E,R:cx.ranks(E,R,queries['new_valid'],truths)
        metrics={}
        ec={}
        reuse={}
        drift_report={}
        evaluation=out/'evaluation'
        evaluation.mkdir(exist_ok=True)
        for arm in ARMS:
            status('evaluate',parent=str(parent),arm=arm)
            m=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
            path=None
            if arm=='none':path=previous.paths(parent)[2];rankpath=rp
            elif arm=='fixed_anchor':
                path=Path('/data/lc/untzh/smallinit_protect126_20261014/seed17/zero/models/anchor_null/epoch-20.pt') if parent==17 else Path('/data/lc/untzh/wn_smallprotect127_20261014/ComplEx/zero/models/answer_anchor/epoch-20.pt')
                rankpath=ap
            elif arm=='fixed_replay' and parent=='WN_ComplEx':
                path=Path('/data/lc/untzh/wn_smallprotect127_20261014/ComplEx/zero/models/full_replay_joint/epoch-20.pt')
                rankpath=path.parent.parent.parent/'evaluation/full_replay_joint.npz'
            equal=False
            if path:
                old=torch.load(path,map_location='cpu',weights_only=True)
                equal=torch.equal(m['E'],old['E']) and torch.equal(m['R'],old['R'])
            reuse[arm]=dict(parameter_equal=equal,reference=str(path) if path else None,not_independent_repeat=equal)
            costs={}
            if equal:
                ranks=base.load(rankpath)
                costs=dict(reused_existing_ranks=True)
            else:
                E,R=m['E'].cuda(),m['R'].cuda()
                ranks={}
                for s in ['old_train','old_valid']:ranks[s],costs[s]=oldrank(E,R,s)
                st=time.monotonic()
                ranks['new_valid']=newrank(E,R)
                costs['new_valid']=dict(seconds=time.monotonic()-st)
            np.savez(evaluation/f'{arm}.npz',**ranks)
            metric=worker.metric(ranks,queries,teacher,split)
            metrics[arm]=metric
            ec[arm]=costs
            drift_report[arm]={s:drift(ranks[s],teacher[s]) for s in ['old_train','old_valid']}
            print('EVALUATED137',parent,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
        for arm in ARMS:
            metrics[arm]['directional_ratios_to_none']=[metrics[arm]['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        report=dict(metrics=metrics,drift=drift_report,reuse=reuse,costs=dict(preparation=prep,training=training,evaluation=ec))
        reports[str(parent)]=report
        base.write(out/'summary.json',report)
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=12,
        equal_compute_envelope=True,all_valid_development=True,go_no_go_requires_strong_fixed_and_random_controls=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
