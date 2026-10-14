import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import copy
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/online_shadow106_20261014')
import experiment106 as online
sys.path.insert(0,'/data/lc/untzh/increment87_20261014')
import experiment87 as existing
sys.path.insert(0,'/data/lc/untzh/query_budget101_20261014')
import experiment101 as query
sys.path.insert(0,'/data/lc/untzh/pair_constraint111_20261014')
import experiment111 as pair
base,core=online.base,online.core
ROOT=Path(__file__).resolve().parent
N=11258
ARMS=['none','absolute','anchor_null','query_uniform']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();status('starting');bundles=[]
    for seed in [43,17,29]:
        source=Path('/data/lc/untzh')/({43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed])
        out=ROOT/f'seed{seed}';out.mkdir();(out/'cache').mkdir();online.configure(out,source);existing.ROOT=out;query.ROOT=out;torch.manual_seed(seed);status('prepare',seed=seed);st=time.monotonic()
        standard=torch.load(source/'cache/initial.pt',map_location='cpu',weights_only=True);initial=copy.deepcopy(standard);initial['E'][N:]*=.25;torch.save(initial,out/'cache/initial.pt')
        tape=base.load(source/'cache/learning.npz');z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');new=z['new'];truths=core.truthmap(z['old'],np.concatenate([new,z['extra']]))
        c=base.load(source/'cache/cohort.npz');teacher=base.load(source/'cache/teacher-train.npz')['old_train']
        data=dict(q=torch.as_tensor(c['q'],device='cuda'),candidates=torch.as_tensor(c['candidates'],device='cuda'),logZ=torch.as_tensor(c['logZ'],device='cuda'),legal_newcount=torch.as_tensor(c['legal_newcount'],device='cuda'),selections=base.load(source/'cache/protection.npz')['selections'],fixed=np.column_stack([teacher[c['ids'],1],np.zeros((len(c['ids']),7))]))
        online.action.extra_data(truths,data)
        spacepath={43:source,17:Path('/data/lc/untzh/anchor_confirm95_20261014'),29:Path('/data/lc/untzh/query_space92_20261014')}[seed]
        space=online.replay.previous.load_space(spacepath/'cache/query-space.pt')
        groups,reps,inverse=np.unique(c['queries'][:,:2],axis=0,return_index=True,return_inverse=True)
        prefix,ps,first=existing.prefix10(initial,new,tape);assert first==220
        calibration,cs=existing.calibrate(initial,prefix,new,tape,data,220)
        costs=dict(preparation_seconds=time.monotonic()-st,prefix_seconds=ps,calibration_seconds=cs,historical_assets_not_free=True,scale=.25,training={})
        for arm in ARMS:
            status('train',seed=seed,arm=arm)
            if arm in ['none','absolute']:report=existing.train(arm,initial,prefix,new,tape,data,calibration,220)
            else:
                anchor=online.OnlineAnchor(initial,prefix,new,tape)
                report=online.absolute.train(arm,initial,prefix,new,tape,data,space,anchor) if arm=='anchor_null' else query.train(arm,initial,prefix,new,tape,data,None,space,anchor,reps,inverse)
                ordinary=torch.load(out/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
                assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
                report.update(ordinary_endpoint_equal=True,shadow_compute_included=True);del anchor
            costs['training'][arm]=report;base.write(out/'costs-running.json',costs)
        bundles.append((seed,out,source,initial,truths,costs))
    # All twelve committed before current VALID evaluation.
    summaries={};full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
    for seed,out,source,initial,truths,costs in bundles:
        online.configure(out,source);status('evaluation_prepare',seed=seed);st=time.monotonic()
        fast=pair.fast43(initial,truths) if seed==43 else pair.fastmodule.FastOld(seed,truths)
        costs['fast_prepare_seconds']=time.monotonic()-st;queries=dict(fast.queries,new_valid=full['new_valid'])
        reference=json.loads((source/'summary.json').read_text())['metrics'] if seed==43 else json.loads(Path('/data/lc/untzh/full_valid102_20261014/summary.json').read_text())['seeds'][str(seed)]['metrics']
        if seed==17:reference['query_uniform']=json.loads(Path('/data/lc/untzh/query_confirm108_20261014/summary.json').read_text())['metrics']['query_uniform']
        evaluation=out/'evaluation';evaluation.mkdir();expanded,expanded_cost=fast.ranks(initial['E'].cuda().double(),initial['R'].cuda().double(),'old_valid')
        low_initial=core.drops(expanded,fast.teacher['old_valid']);base.write(evaluation/'low-expanded-initial.json',dict(drops=low_initial,cost=expanded_cost));metrics={};ec={}
        for arm in ARMS:
            status('evaluate',seed=seed,arm=arm);m=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda();states={};cost={}
            for split in ['old_train','old_valid']:states[split],cost[split]=fast.ranks(E,R,split)
            t=time.monotonic();states['new_valid']=base.ranks(E,R,full['new_valid'],truths,arm=arm,split='new_valid');cost['new_valid']=dict(seconds=time.monotonic()-t);np.savez(evaluation/f'{arm}.npz',**states)
            metric={s:base.metrics(states[s],q) for s,q in queries.items()}
            for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],fast.teacher[split])
            metric['directional_ratios_to_low_none']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS] if arm!='none' else [1.,1.,1.]
            metric['directional_ratios_to_standard_none']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=metric;ec[arm]=cost;base.write(evaluation/'metrics-running.json',metrics)
            print('EVALUATED116',seed,arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],metric['directional_ratios_to_low_none'],flush=True)
        costs['evaluation']=ec;summaries[str(seed)]=dict(metrics=metrics,standard_controls=reference,low_initial_drops=low_initial,costs=costs);base.write(out/'summary.json',summaries[str(seed)])
    base.write(ROOT/'summary.json',dict(seeds=summaries,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
