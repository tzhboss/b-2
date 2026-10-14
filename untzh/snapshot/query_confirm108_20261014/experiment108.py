import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/query_budget101_20261014')
import experiment101 as query
sys.path.insert(0,'/data/lc/untzh/online_shadow106_20261014')
import experiment106 as online
sys.path.insert(0,'/data/lc/untzh/fast_old_eval107_20261014')
import experiment107 as fastmodule
base,core=online.base,online.core
ROOT=Path(__file__).resolve().parent
ARMS=['query_random_pool','query_uniform']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();status('prepare')
    assert json.loads((fastmodule.ROOT/'summary.json').read_text())['all_equal']
    source=Path('/data/lc/untzh/increment87_20261014');online.configure(ROOT,source);query.ROOT=ROOT
    initial,prefix,new,tape,truths,data,ps=online.source17.prepare()
    space=online.replay.previous.load_space('/data/lc/untzh/anchor_confirm95_20261014/cache/query-space.pt')
    c=base.load(source/'cache/cohort.npz')
    groups,representatives,inverse=np.unique(c['queries'][:,:2],axis=0,return_index=True,return_inverse=True)
    assert np.array_equal(c['q'],c['q'][representatives[inverse]])
    (ROOT/'cache').mkdir();np.savez(ROOT/'cache/query-groups.npz',groups=groups,representatives=representatives,inverse=inverse)
    preparation=time.monotonic()-tick;training={}
    ordinary=torch.load(source/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
    for arm in ARMS:
        status('train',arm=arm);setup=time.monotonic();anchor=online.OnlineAnchor(initial,prefix,new,tape)
        setup_seconds=time.monotonic()-setup
        training[arm]=query.train(arm,initial,prefix,new,tape,data,None,space,anchor,representatives,inverse)
        assert anchor.step==440
        assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
        training[arm].update(online_setup_seconds=setup_seconds,shadow_compute_included=True,ordinary_endpoint_equal=True)
        base.write(ROOT/'costs-running.json',dict(preparation_seconds=preparation,training=training))
        del anchor
    # Both models committed before any VALID evaluation.
    status('evaluation_prepare');fast=fastmodule.FastOld(17,truths)
    queries=dict(fast.queries,new_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['new_valid'])
    reference=json.loads(Path('/data/lc/untzh/full_valid102_20261014/summary.json').read_text())['seeds']['17']['metrics']
    evaluation=ROOT/'evaluation';evaluation.mkdir();np.savez(evaluation/'queries.npz',**queries)
    metrics={};evalcost={};evaluation_tick=time.monotonic()
    for arm in ARMS:
        status('evaluate',arm=arm);m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        E,R=m['E'].cuda(),m['R'].cuda();states={};cost={}
        for split in ['old_train','old_valid']:states[split],cost[split]=fast.ranks(E,R,split)
        start=time.monotonic();states['new_valid']=base.ranks(E,R,queries['new_valid'],truths,arm=arm,split='new_valid')
        cost['new_valid']=dict(seconds=time.monotonic()-start)
        np.savez(evaluation/f'{arm}.npz',**states)
        metric={s:base.metrics(states[s],q) for s,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],fast.teacher[split])
        metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        metrics[arm]=metric;evalcost[arm]=cost;base.write(evaluation/'metrics-running.json',metrics)
        print('EVALUATED108',arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
    summary=dict(seed=17,metrics=metrics,controls_full_valid=reference,costs=dict(preparation_seconds=preparation,query_groups=len(groups),cohort_answers=len(c['queries']),training=training,fast_evaluation_preparation=fast.preparation,evaluation=evalcost,evaluation_seconds=time.monotonic()-evaluation_tick,total_seconds=time.monotonic()-tick,historical_assets_not_free=True))
    base.write(ROOT/'summary.json',summary);status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
