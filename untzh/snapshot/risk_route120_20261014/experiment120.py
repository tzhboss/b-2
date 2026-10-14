import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/lowinit_confirm116_20261014')
import experiment116 as low
online,base,core,pair=low.online,low.base,low.core,low.pair
ROOT=Path(__file__).resolve().parent
N=11258
ARMS=['uniform_safe','pressure_safe','tree_safe','random_safe']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def selections(prediction,seed):
    safe=np.flatnonzero(prediction['current']<=10);size=int(np.ceil(.05*len(safe)))
    pools=dict(pressure_safe=safe[np.argsort(-prediction['pressure128'][safe],kind='stable')[:size]],
               tree_safe=safe[np.argsort(-prediction['tree_current'][safe],kind='stable')[:size]],
               random_safe=np.random.default_rng(1200000+seed).choice(safe,size,replace=False))
    chosen={}
    for arm in ARMS:
        rng=np.random.default_rng(1201000+seed);rows=[]
        for j in range(220):
            if arm=='uniform_safe':row=rng.choice(safe,32,replace=False)
            else:
                first=rng.choice(pools[arm],26,replace=False)
                available=safe[~np.isin(safe,first)]
                row=np.concatenate([first,rng.choice(available,6,replace=False)])
            assert len(np.unique(row))==32;rows.append(row)
        chosen[arm]=np.stack(rows)
    return chosen,pools,safe

def train(arm,initial,prefix,new,tape,data,space,chosen,out,ordinary):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix)
    anchor=online.OnlineAnchor(initial,prefix,new,tape);records=[]
    for j in range(220,440):
        ids=chosen[j-220];loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        penalty,_=online.action.penalty(E,data,ids,'absolute_full');gp=torch.autograd.grad(penalty,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchor[j-220];difference=(E.detach()[N:]-reference).double()
        projected=online.prior.project(difference,space)
        with torch.no_grad():E[N:].copy_(reference+projected.to(E.dtype))
        actual=(E.detach()[N:]-reference).double()
        changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,gradient_ratio=alpha*pn/bn if bn else 0.,
                            anchor_norm=float(actual.norm()),cross_max=float(changes.abs().max())))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
    target=out/'models'/arm;target.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),target/'epoch-20.pt')
    report=dict(seconds=time.monotonic()-tick,steps=220,shadow_compute_included=True,
                ordinary_endpoint_equal=True,mean_gradient_ratio=float(np.mean([v['gradient_ratio'] for v in records])),
                max_cross_score_difference=max(v['cross_max'] for v in records),final_anchor_norm=records[-1]['anchor_norm'])
    base.write(target/'steps.json',records);base.write(target/'training.json',report)
    return report

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();bundles=[];status('prepare')
    z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');new=z['new'];truths=core.truthmap(z['old'],np.concatenate([new,z['extra']]))
    for seed in [43,17,29]:
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
        previous=Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{seed}');out=ROOT/f'seed{seed}'
        (out/'cache').mkdir(parents=True,exist_ok=True);online.configure(out,src);st=time.monotonic()
        initial=torch.load(previous/'cache/initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(previous/'cache/prefix10.pt',map_location='cpu',weights_only=True)
        ordinary=torch.load(previous/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
        tape=base.load(src/'cache/learning.npz');c=base.load(src/'cache/cohort.npz')
        prediction=base.load(f'/data/lc/untzh/risk_generalization119_20261014/cache/{seed}-low-old_train-predictions.npz')
        sample=base.load(f'/data/lc/untzh/risk_generalization119_20261014/cache/{seed}-old_train-sample.npz')
        assert np.array_equal(sample['ids'],c['ids']) and np.array_equal(sample['queries'],c['queries'])
        chosen,pools,safe=selections(prediction,seed)
        np.savez(out/'cache/selections.npz',safe=safe,**chosen,**{k+'_pool':v for k,v in pools.items()})
        data=dict(q=torch.as_tensor(c['q'],device='cuda'))
        # Only the exact legal mask is needed by absolute_full; omit capacity/rank actions.
        legal=np.ones((len(c['ids']),1876),dtype=bool)
        for i,(h,r,g) in enumerate(c['queries']):
            legal[i,[x-N for x in truths[int(h),int(r)] if x>=N]]=False
        data['legal']=torch.as_tensor(legal,device='cuda')
        spacepath={43:src,17:Path('/data/lc/untzh/anchor_confirm95_20261014'),29:Path('/data/lc/untzh/query_space92_20261014')}[seed]
        space=online.replay.previous.load_space(spacepath/'cache/query-space.pt')
        costs=dict(preparation_seconds=time.monotonic()-st,safe_answers=len(safe),pool_answers=len(pools['pressure_safe']),training={},historical_detection_cost_not_free=True)
        for arm in ARMS:
            status('train',seed=seed,arm=arm)
            costs['training'][arm]=train(arm,initial,prefix,new,tape,data,space,chosen[arm],out,ordinary)
            base.write(out/'costs-running.json',costs)
        bundles.append((seed,src,previous,out,initial,costs))
    # All twelve new models committed before current VALID evaluation; TRAIN-only routing.
    full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz');reports={}
    for seed,src,previous,out,initial,costs in bundles:
        online.configure(out,src);status('evaluate_prepare',seed=seed)
        fast=pair.fast43(initial,truths) if seed==43 else pair.fastmodule.FastOld(seed,truths)
        queries=dict(fast.queries,new_valid=full['new_valid'])
        reference=json.loads((previous/'summary.json').read_text(encoding='utf8'))['metrics'];metrics={};ec={}
        evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True);st=time.monotonic()
        for arm in ARMS:
            status('evaluate',seed=seed,arm=arm)
            model=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=model['E'].cuda(),model['R'].cuda();states={};cost={}
            for split in ['old_train','old_valid']:states[split],cost[split]=fast.ranks(E,R,split)
            start=time.monotonic();states['new_valid']=base.ranks(E,R,full['new_valid'],truths,arm=arm,split='new_valid');cost['new_valid']=dict(seconds=time.monotonic()-start)
            np.savez(evaluation/f'{arm}.npz',**states)
            m={split:base.metrics(states[split],q) for split,q in queries.items()}
            for split in ['old_train','old_valid']:m[split]['drops']=core.drops(states[split],fast.teacher[split])
            m['directional_ratios_to_low_none']=[m['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=m;ec[arm]=cost;base.write(evaluation/'metrics-running.json',metrics)
            print('EVALUATED120',seed,arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_train']['drops']['filtered_train_truths']['drops'],m['old_valid']['drops']['filtered_train_truths']['drops'],m['directional_ratios_to_low_none'],flush=True)
        costs.update(evaluation=ec,evaluation_seconds=time.monotonic()-st)
        reports[str(seed)]=dict(metrics=metrics,reused_controls=reference,costs=costs);base.write(out/'summary.json',reports[str(seed)])
    base.write(ROOT/'summary.json',dict(seeds=reports,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
