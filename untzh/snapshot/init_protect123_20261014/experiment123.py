import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/lowinit_confirm116_20261014')
import experiment116 as previous
online,base,core,pair,existing=previous.online,previous.base,previous.core,previous.pair,previous.existing
ROOT=Path(__file__).resolve().parent
N=11258
INITS=['neighbor_mean','query_mean']
ARMS=['absolute','anchor_null']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();status('prepare');bundles=[]
    z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');new=z['new'];truths=core.truthmap(z['old'],np.concatenate([new,z['extra']]))
    for seed in [43,17,29]:
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
        tape=base.load(src/'cache/learning.npz');c=base.load(src/'cache/cohort.npz')
        for init_name in INITS:
            out=ROOT/f'seed{seed}'/init_name;out.mkdir(parents=True,exist_ok=True)
            online.configure(out,src);existing.ROOT=out;st=time.monotonic()
            statepath=Path(f'/data/lc/untzh/train_init121_20261014/seed{seed}/models/{init_name}')
            initial=torch.load(statepath/'initial.pt',map_location='cpu',weights_only=True)
            prefix=torch.load(statepath/'prefix10.pt',map_location='cpu',weights_only=True)
            ordinary=torch.load(statepath/'epoch-20.pt',map_location='cpu',weights_only=True)
            data=dict(q=torch.as_tensor(c['q'],device='cuda'),candidates=torch.as_tensor(c['candidates'],device='cuda'),
                      logZ=torch.as_tensor(c['logZ'],device='cuda'),legal_newcount=torch.as_tensor(c['legal_newcount'],device='cuda'),
                      selections=base.load(src/'cache/protection.npz')['selections'])
            legal=np.ones((len(c['ids']),1876),dtype=bool)
            for i,(h,r,g) in enumerate(c['queries']):legal[i,[x-N for x in truths[int(h),int(r)] if x>=N]]=False
            data['legal']=torch.as_tensor(legal,device='cuda')
            calibration,cs=existing.calibrate(initial,prefix,new,tape,data,220)
            spacepath={43:src,17:Path('/data/lc/untzh/anchor_confirm95_20261014'),29:Path('/data/lc/untzh/query_space92_20261014')}[seed]
            space=online.replay.previous.load_space(spacepath/'cache/query-space.pt')
            costs=dict(preparation_seconds=time.monotonic()-st,calibration_seconds_included=cs,historical_prefix_reused=True,training={})
            for arm in ARMS:
                status('train',seed=seed,init=init_name,arm=arm)
                if arm=='absolute':report=existing.train(arm,initial,prefix,new,tape,data,calibration,220)
                else:
                    anchor=online.OnlineAnchor(initial,prefix,new,tape)
                    report=online.absolute.train(arm,initial,prefix,new,tape,data,space,anchor)
                    assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
                    report.update(shadow_compute_included=True,ordinary_endpoint_equal=True)
                costs['training'][arm]=report;base.write(out/'costs-running.json',costs)
            bundles.append((seed,init_name,src,out,initial,costs))
    # All twelve new endpoints committed before current VALID read.
    full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz');reports={}
    for seed,init_name,src,out,initial,costs in bundles:
        online.configure(out,src);status('evaluation_prepare',seed=seed,init=init_name)
        fast=pair.fast43(initial,truths) if seed==43 else pair.fastmodule.FastOld(seed,truths)
        queries=dict(fast.queries,new_valid=full['new_valid'])
        ordinary=json.loads(Path(f'/data/lc/untzh/train_init121_20261014/seed{seed}/summary.json').read_text(encoding='utf8'))['metrics'][init_name]
        low=json.loads(Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{seed}/summary.json').read_text(encoding='utf8'))['metrics']
        metrics={};ec={};directory=out/'evaluation';directory.mkdir(exist_ok=True);st=time.monotonic()
        for arm in ARMS:
            status('evaluate',seed=seed,init=init_name,arm=arm)
            model=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=model['E'].cuda(),model['R'].cuda();states={};cost={}
            for split in ['old_train','old_valid']:states[split],cost[split]=fast.ranks(E,R,split)
            s=time.monotonic();states['new_valid']=base.ranks(E,R,full['new_valid'],truths,arm=arm,split='new_valid');cost['new_valid']=dict(seconds=time.monotonic()-s);np.savez(directory/f'{arm}.npz',**states)
            m={split:base.metrics(states[split],q) for split,q in queries.items()}
            for split in ['old_train','old_valid']:m[split]['drops']=core.drops(states[split],fast.teacher[split])
            m['directional_ratios_to_own_none']=[m['new_valid'][d]['filtered_train_truths']['mrr']/ordinary['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            m['directional_ratios_to_low_none']=[m['new_valid'][d]['filtered_train_truths']['mrr']/low['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=m;ec[arm]=cost;base.write(directory/'metrics-running.json',metrics)
            print('EVALUATED123',seed,init_name,arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_train']['drops']['filtered_train_truths']['drops'],m['old_valid']['drops']['filtered_train_truths']['drops'],m['directional_ratios_to_low_none'],flush=True)
        costs.update(evaluation=ec,evaluation_seconds=time.monotonic()-st)
        report=dict(metrics=metrics,reused_own_none=ordinary,reused_low_controls=low,costs=costs)
        reports[f'{seed}-{init_name}']=report;base.write(out/'summary.json',report)
    base.write(ROOT/'summary.json',dict(cases=reports,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
