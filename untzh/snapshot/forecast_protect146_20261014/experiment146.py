import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/trajectory_risk144_20261014')
import experiment144 as risk
sys.path.insert(0,'/data/lc/untzh/init_protect123_20261014')
import experiment123 as controls
worker,base,core,cx=risk.worker,risk.base,risk.core,risk.cx
ROOT=Path(__file__).resolve().parent
N,TOTAL=11258,13134
ARMS=['none','fixed_anchor','fixed_replay','forecast_anchor','current_anchor','random_matched']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def sync():torch.cuda.synchronize();return time.monotonic()

def assets(seed,src,truths,out):
    begin=sync()
    c=base.load(src/'cache/cohort.npz')
    legal=np.ones((len(c['ids']),TOTAL-N),bool)
    for i,(h,r,g) in enumerate(c['queries']):
        legal[i,[v-N for v in truths[int(h),int(r)] if v>=N]]=False
    cached_selections=base.load(src/'cache/protection.npz')['selections']
    anchor=dict(q=torch.as_tensor(c['q'],device='cuda'),legal=torch.as_tensor(legal,device='cuda'),
                selections=cached_selections[:220])
    spacepath={43:src,17:Path('/data/lc/untzh/anchor_confirm95_20261014'),29:Path('/data/lc/untzh/query_space92_20261014')}[seed]
    space=controls.online.replay.previous.load_space(spacepath/'cache/query-space.pt')
    replaypath=src/'cache/full-replay.npz' if seed==43 else Path(f'/data/lc/untzh/full_replay105_20261014/seed{seed}/cache/full-replay.npz')
    z=base.load(replaypath)
    translate=dict(qt='qtail',qh='qhead',mt='mask_tail',mh='mask_head',dt='denominator_tail',dh='denominator_head')
    replay={k:torch.as_tensor(z[v],device='cuda') for k,v in translate.items()}
    predictionpath=Path('/data/lc/untzh/trajectory_risk144_20261014/17') if seed==17 else Path(f'/data/lc/untzh/trajectory_confirm145_20261014/seed{seed}')
    pred=base.load(predictionpath/'old_train-predictions.npz')
    pos=np.searchsorted(pred['ids'],c['ids'])
    assert np.array_equal(pred['ids'][pos],c['ids'])
    forecast=pred['stationary_mean_rank'][pos]>10
    current=pred['current_rank'][pos]>10
    fg=forecast[anchor['selections']]
    cg=current[anchor['selections']]
    rng=np.random.default_rng(1460017)
    rg=np.zeros_like(fg)
    for i,count in enumerate(fg.sum(1)):rg[i,rng.choice(fg.shape[1],int(count),replace=False)]=True
    assert np.array_equal(fg.sum(1),rg.sum(1))
    gates=dict(forecast_anchor=fg,current_anchor=cg,random_matched=rg)
    np.savez(out/'gates-frozen.npz',cohort_ids=c['ids'],forecast_query=forecast,current_query=current,**gates)
    report=dict(seconds=sync()-begin,forecast_queries=int(forecast.sum()),current_queries=int(current.sum()),cohort=len(forecast),
        foreground_draws=int(fg.sum()),all_query_draws=int(fg.size),forecast_trigger_steps=int(fg.any(1).sum()),
        original_cached_selection_rows=len(cached_selections),used_selection_rows=len(fg),
        current_trigger_steps=int(cg.any(1).sum()),random_step_counts_equal=True,original_selections_unchanged=True,
        prediction_reference=str(predictionpath),cached_forecast_not_free=True,seed43_historical_forecast_timing_unknown=seed==43)
    base.write(out/'preparation.json',report)
    return anchor,replay,space,gates,report

def train(arm,initial,prefix,ordinary,new,tape,anchor,replay,space,gates,out):
    E,R,opt=worker.model(initial,N,prefix)
    se,sr,so=worker.model(initial,N,prefix)
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    assert len(tape['epochs'])-start==220
    masks={name:torch.as_tensor(value,device='cuda') for name,value in gates.items()}
    costs={k:0. for k in ['learning','all_protection_gradients','update','ordinary_shadow','projection','recording']}
    records=[]
    begin=sync()
    for j in range(start,len(tape['epochs'])):
        k=j-start
        tick=sync()
        gl=torch.autograd.grad(base.learning(E,R,new,tape,j),E)[0]
        costs['learning']+=sync()-tick
        tick=sync()
        chosen=anchor['selections'][k]
        values=F.softplus(anchor['q'][chosen]@E[N:].double().T).masked_fill(~anchor['legal'][chosen],0)
        query_loss=values.sum(1)/anchor['legal'][chosen].sum(1)
        ga=torch.autograd.grad(query_loss.mean(),E,retain_graph=True)[0]
        sparse={}
        for name in ['forecast_anchor','current_anchor','random_matched']:
            sparse[name]=torch.autograd.grad((query_loss*masks[name][k]).mean(),E,retain_graph=True)[0]
        tail=F.softplus(replay['qt'][k]@E[N:].double().T).masked_fill(~replay['mt'][k],0).sum(1)/replay['dt'][k]
        head=F.softplus(replay['qh'][k]@E[N:].double().T).masked_fill(~replay['mh'][k],0).sum(1)/replay['dh'][k]
        gr=torch.autograd.grad(.5*(tail+head).mean(),E)[0]
        costs['all_protection_gradients']+=sync()-tick
        tick=sync()
        gp=ga if arm=='fixed_anchor' else gr if arm=='fixed_replay' else sparse[arm] if arm in sparse else torch.zeros_like(gl)
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True)
        E.grad=gl+alpha*gp
        opt.step()
        costs['update']+=sync()-tick
        tick=sync()
        so.zero_grad(set_to_none=True)
        base.learning(se,sr,new,tape,j).backward()
        so.step()
        costs['ordinary_shadow']+=sync()-tick
        tick=sync()
        ref=se.detach()[N:]
        accepted=cx.source.space_module.project((E.detach()[N:]-ref).double(),space)
        if arm not in ['none','fixed_replay']:
            with torch.no_grad():E[N:].copy_(ref+accepted.to(E.dtype))
        costs['projection']+=sync()-tick
        tick=sync()
        delta=(E.detach()[N:]-ref).double()
        changes=(space['vectors']*delta[space['rows']]).sum(1)
        assert bool(torch.isfinite(E).all())
        count=int(masks[arm][k].sum()) if arm in masks else 32 if arm in ['fixed_anchor','fixed_replay'] else 0
        records.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,gp_norm=pn,selected_queries=count,
            delta_norm=float(delta.norm()),cross_max=float(changes.abs().max())))
        costs['recording']+=sync()-tick
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    dest=out/'models'/arm
    dest.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),dest/'epoch-20.pt')
    base.write(dest/'steps.json',records)
    report=dict(seconds=sync()-begin,phase_costs=costs,steps=len(records),
        nonzero_steps=sum(r['gp_norm']>0 for r in records),mean_ratio=float(np.mean([r['ratio'] for r in records])),
        selected_query_draws=sum(r['selected_queries'] for r in records),query_draw_fraction=float(np.mean([r['selected_queries']/32 for r in records])),
        protection_gradient_norm_sum=float(sum(r['gp_norm'] for r in records)),final_displacement_norm=records[-1]['delta_norm'],
        max_cross_score_difference=max(r['cross_max'] for r in records),shadow_equal=True,frozen_equal=True,
        all_arms_equal_gradient_envelope=True,not_equal_cumulative_strength=True)
    base.write(dest/'training.json',report)
    return report

def drift(state,teacher):
    return {name:dict(mean_rank_delta=float((state[:,col]-teacher[:,col]).mean()),
        mean_log_rank_ratio=float(np.log(state[:,col]/teacher[:,col]).mean()),worsened=int((state[:,col]>teacher[:,col]).sum()),
        mean_rr_delta=float((1/state[:,col]-1/teacher[:,col]).mean())) for name,col in [('raw',0),('filtered_train_truths',1)]}

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    _,_,_,new,_,truths=risk.training_assets(17)
    bundles=[]
    for seed in [17,43,29]:
        status('prepare',seed=seed)
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
        path=Path(f'/data/lc/untzh/train_init121_20261014/seed{seed}/models/zero')
        initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(path/'prefix10.pt',map_location='cpu',weights_only=True)
        ordinary=torch.load(path/'epoch-20.pt',map_location='cpu',weights_only=True)
        tape=base.load(src/'cache/learning.npz')
        out=ROOT/f'seed{seed}'
        out.mkdir(parents=True,exist_ok=True)
        anchor,replay,space,gates,prep=assets(seed,src,truths,out)
        training={}
        for arm in ARMS:
            status('train',seed=seed,arm=arm)
            training[arm]=train(arm,initial,prefix,ordinary,new,tape,anchor,replay,space,gates,out)
            print('TRAIN146',seed,arm,training[arm],flush=True)
        bundles.append((seed,src,path,initial,out,prep,training))
    base.write(ROOT/'committed-models.json',dict(new_training_configs=18,all_endpoints_before_current_valid=True,
               gate_train_only=True,equal_compute_envelope=True,no_test=True))
    full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
    reports={}
    for seed,src,path,initial,out,prep,training in bundles:
        status('evaluation_prepare',seed=seed)
        fast=controls.pair.fast43(initial,truths) if seed==43 else controls.pair.fastmodule.FastOld(seed,truths)
        queries=dict(fast.queries,new_valid=full['new_valid'])
        metrics={}
        reuse={}
        costs={}
        drifts={}
        directory=out/'evaluation'
        directory.mkdir(exist_ok=True)
        for arm in ARMS:
            status('evaluate',seed=seed,arm=arm)
            model=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
            ref=None
            if arm=='none':ref=path/'epoch-20.pt';rankpath=path.parent.parent/'evaluation/zero.npz'
            elif arm=='fixed_anchor':
                ref=Path(f'/data/lc/untzh/smallinit_protect126_20261014/seed{seed}/zero/models/anchor_null/epoch-20.pt')
                rankpath=ref.parent.parent.parent/'evaluation/anchor_null.npz'
            equal=False
            if ref:
                control=torch.load(ref,map_location='cpu',weights_only=True)
                equal=torch.equal(model['E'],control['E']) and torch.equal(model['R'],control['R'])
            reuse[arm]=dict(parameter_equal=equal,reference=str(ref) if ref else None)
            if equal:
                states=base.load(rankpath)
                cost=dict(existing_ranks_reused=True)
            else:
                E,R=model['E'].cuda(),model['R'].cuda()
                states={}
                cost={}
                for split in ['old_train','old_valid']:states[split],cost[split]=fast.ranks(E,R,split)
                tick=time.monotonic()
                states['new_valid']=base.ranks(E,R,full['new_valid'],truths)
                cost['new_valid']=dict(seconds=time.monotonic()-tick)
            np.savez(directory/f'{arm}.npz',**states)
            metric={split:base.metrics(states[split],q) for split,q in queries.items()}
            for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],fast.teacher[split])
            metrics[arm]=metric
            costs[arm]=cost
            drifts[arm]={split:drift(states[split],fast.teacher[split]) for split in ['old_train','old_valid']}
            print('EVALUATED146',seed,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],
                metric['old_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
        for arm in ARMS:
            metrics[arm]['directional_ratios_to_none']=[metrics[arm]['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        def values(arm):
            m=metrics[arm]
            return (m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'])
        candidate=values('forecast_anchor')
        strong=[values(a) for a in ['fixed_anchor','fixed_replay','random_matched']]
        best=(max(v[0] for v in strong),max(v[1] for v in strong),min(v[2] for v in strong))
        joint=bool(candidate[0]>=best[0] and candidate[1]>=best[1] and candidate[2]<=best[2] and candidate!=best)
        directions=bool(min(metrics['forecast_anchor']['directional_ratios_to_none'])>=.98)
        report=dict(metrics=metrics,drift=drifts,reuse=reuse,costs=dict(preparation=prep,training=training,evaluation=costs),
            go_checks=dict(joint_beats_strong_and_random=joint,direction_floor_pass=directions),
            forecast_candidates=list(candidate),strong_envelope=list(best))
        reports[str(seed)]=report
        base.write(out/'summary.json',report)
    eligible=all(all(r['go_checks'].values()) for r in reports.values())
    base.write(ROOT/'summary.json',dict(seeds=reports,total_seconds=time.monotonic()-begin,new_training_configs=18,new_parent_trainings=0,
        decision='GO performance candidate; cost review required' if eligible else 'NO-GO stop forecast Top10 trigger; no threshold repair',
        all_valid_development=True,no_test=True,gate_train_only=True,equal_compute_envelope=True))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
