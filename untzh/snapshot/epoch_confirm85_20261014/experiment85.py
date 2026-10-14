import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/epoch_guard84_20261014')
import experiment84 as prev
core=prev.core;base=prev.base;ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/blend_confirm80_20261014')

def prepare(seed,old,new):
    tick=time.monotonic();folder=SOURCE/f'seed{seed}';z=base.load(folder/'cache/cohort.npz')
    d={k:torch.as_tensor(z[k],device='cuda') for k in ['q','candidates','logZ','legal_newcount']}
    d['selections']=base.load(folder/'cache/protection.npz')['selections']
    initial=torch.load(core.SRC/f'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s{seed}/initial.pt',map_location='cpu',weights_only=True)
    d['gold']=(d['q']*initial['E'][z['queries'][:,2]].cuda().double()).sum(-1)
    teacher=base.load(folder/'cache/teacher-train.npz')['old_train'];tr=core.truthmap(old,new)
    rng=np.random.default_rng(840000+seed);rows=rng.choice(core.SIZE,1024,replace=False)
    oldmask=np.ones((1024,core.TOTAL-core.N),bool)
    for i,(h,r,g) in enumerate(z['queries'][rows]):oldmask[i,[c-core.N for c in tr[int(h),int(r)] if c>=core.N]]=False
    masks=[(new[:,0]<core.N)&(new[:,2]>=core.N),(new[:,0]>=core.N)&(new[:,2]<core.N),(new[:,0]>=core.N)&(new[:,2]>=core.N)]
    newqueries=np.concatenate([new[rng.choice(np.flatnonzero(mask),256,replace=False)] for mask in masks])
    newmask=np.ones((len(newqueries),core.TOTAL),bool)
    for i,(h,r,g) in enumerate(newqueries):newmask[i,[c for c in tr[int(h),int(r)] if c!=int(g)]]=False
    pd=dict(old_q=d['q'][rows],old_gold=d['gold'][rows],old_rank=torch.as_tensor(teacher[z['ids'][rows],1],device='cuda'),
        old_legal=torch.as_tensor(oldmask,device='cuda'),new_q=torch.as_tensor(newqueries,device='cuda'),new_legal=torch.as_tensor(newmask,device='cuda'))
    np.savez(prev.ROOT/'probe-cache.npz',old_cohort_rows=rows,old_legal=oldmask,new_queries=newqueries,new_legal=newmask)
    prefix=torch.load(folder/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    tape=base.load(folder/'cache/learning.npz');coef=prev.read(folder/'calibration.json')['coefficients']['absolute']
    return initial,prefix,tape,d,pd,tr,coef,dict(seconds=time.monotonic()-tick,cache_source=str(folder/'cache'),old_probe=1024,new_probe_per_direction=256)

def evaluate(seed,old,new,tr):
    tick=time.monotonic();folder=SOURCE/f'seed{seed}'
    A=base.load(folder/'evaluation/teacher.npz');valid=base.load(core.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    p=prev.read(folder/'evaluation/metrics.json');report=dict(none=p['none'],absolute_fixed=p['absolute']);known=dict(none=folder/'models/none/epoch-20.pt',absolute_fixed=folder/'models/absolute/epoch-20.pt')
    out=prev.ROOT/'evaluation';out.mkdir();aliases={}
    for arm in prev.ARMS:
        path=prev.ROOT/'models'/arm/'epoch-20.pt';m=torch.load(path,map_location='cpu',weights_only=True);alias=None
        for name,kpath in known.items():
            k=torch.load(kpath,map_location='cpu',weights_only=True)
            if torch.equal(m['E'],k['E']) and torch.equal(m['R'],k['R']):alias=name;break
        if alias:report[arm]=copy.deepcopy(report[alias]);aliases[arm]=alias;continue
        E,R=m['E'].cuda(),m['R'].cuda();qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={key:base.ranks(E,R,q,tr,arm=arm,split=key) for key,q in qs.items()};np.savez(out/f'{arm}.npz',**states)
        report[arm]={key:base.metrics(states[key],q) for key,q in qs.items()}
        for key in ['old_train','old_valid']:report[arm][key]['drops']=core.drops(states[key],A[key])
        known[arm]=path;core.progress('evaluated',seed=seed,arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm,m in report.items():m['directional_ratios']=[m['new_valid'][x]['filtered_train_truths']['mrr']/report['none']['new_valid'][x]['filtered_train_truths']['mrr'] for x in core.DIRECTIONS]
    core.dump('evaluation/metrics.json',report);core.dump('evaluation/reused-equal-models.json',aliases)
    return report,time.monotonic()-tick

def run(seed):
    tick=time.monotonic();torch.manual_seed(seed)
    prev.ROOT=ROOT/f'seed{seed}';prev.ROOT.mkdir();core.ROOT=prev.ROOT;base.ROOT=prev.ROOT
    z=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=z['old'],z['new']
    initial,prefix,t,d,pd,tr,c,pc=prepare(seed,old,new)
    sr,ss=prev.shadow(initial,prefix,new,t,d,pd,c);costs=dict(preparation=pc,shadow_seconds=ss,training={})
    for policy in ['local','cumulative']:
        costs['training'][policy]=prev.adaptive(policy,initial,prefix,new,t,d,pd,c,sr)
        seq=costs['training'][policy]['selected_sequence']
        costs['training'][policy+'_mean']=prev.scheduled(policy+'_mean',[np.mean(seq)]*10,initial,prefix,new,t,d,c)
        costs['training'][policy+'_shuffle']=prev.scheduled(policy+'_shuffle',np.random.default_rng(840000+seed).permutation(seq),initial,prefix,new,t,d,c)
        core.dump('costs-running.json',costs)
    report,es=evaluate(seed,old,new,tr);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs));core.progress('complete',seed=seed,seconds=costs['total_seconds']);return report,costs

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();reports={};costs={}
    for seed in [29,43]:reports[str(seed)],costs[str(seed)]=run(seed)
    core.ROOT=ROOT;core.dump('summary.json',dict(metrics=reports,costs=costs,total_seconds=time.monotonic()-tick));core.progress('complete')

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
