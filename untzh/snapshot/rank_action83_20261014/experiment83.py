import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import sys
import time
from pathlib import Path
import copy
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/action_guard82_20261014')
import experiment82 as prev
core=prev.core;base=prev.base;ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;base.ROOT=ROOT
ARMS=['absolute_guard','absolute_meanmatched']

@torch.no_grad()
def rank_probes(prefix,R,probes,tr):
    tick=time.monotonic();E=prefix['E'].cuda().double();R=R.cuda().double();rng=np.random.default_rng(830017);saved={}
    for i,z in enumerate(probes):
        p=z['p'];scores=(E[p[:,0]]*R[p[:,1]])@E.T
        for row,(h,r,t) in enumerate(p.cpu().numpy()):scores[row,list(tr[int(h),int(r)])]=-torch.inf
        top=scores.topk(64,dim=-1).indices.cpu().numpy();pool=[]
        for row,(h,r,t) in enumerate(p.cpu().numpy()):
            forbidden=set(tr[int(h),int(r)])|set(map(int,top[row]));extra=[]
            while len(extra)<64:
                value=int(rng.integers(core.TOTAL))
                if value not in forbidden:extra.append(value);forbidden.add(value)
            pool.append(np.concatenate([top[row],extra]))
        ids=np.array(pool,dtype=np.int64);z['candidates']=torch.as_tensor(ids,device='cuda')
        assert all(len(np.unique(row))==128 for row in ids)
        saved[f'direction{i}_candidates']=ids
    np.savez(ROOT/'rank-probe-candidates.npz',**saved)
    return dict(seconds=time.monotonic()-tick,candidates=128,top64_source='prefix10 TRAIN-only filtered',uniform64='TRAIN-only legal, unique')

@torch.no_grad()
def rank_cost(E,R,probes):
    Ed=E.double();Rd=R.double();costs=[]
    for z in probes:
        h,r,t=z['p'].unbind(1);q=Ed[h]*Rd[r]
        gold=(q*Ed[t]).sum(-1);scores=(q[:,None]*Ed[z['candidates']]).sum(-1)
        ranks=1+(scores>gold[:,None]).sum(-1)
        costs.append(1./(1./ranks.double()).mean())
    return torch.stack(costs)

def evaluate(old,new):
    tick=time.monotonic();tr=core.truthmap(old,new)
    valid=base.load(core.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    A=base.load(core.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    prior=json.loads(Path('/data/lc/untzh/risk58_20261013/evaluation/metrics.json').read_text())
    report=dict(none=prior['new_only'],absolute_fixed=prior['uniform']);aliases={};out=ROOT/'evaluation';out.mkdir()
    known={'absolute_fixed':Path('/data/lc/untzh/risk58_20261013/models/uniform/epoch-20.pt')}
    for arm in ARMS:
        path=ROOT/'models'/arm/'epoch-20.pt';m=torch.load(path,map_location='cpu',weights_only=True)
        alias=None
        for name,kpath in known.items():
            k=torch.load(kpath,map_location='cpu',weights_only=True)
            if torch.equal(m['E'],k['E']) and torch.equal(m['R'],k['R']):alias=name;break
        if alias:
            report[arm]=copy.deepcopy(report[alias]);aliases[arm]=alias;continue
        E,R=m['E'].cuda(),m['R'].cuda();qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={key:base.ranks(E,R,q,tr,arm=arm,split=key) for key,q in qs.items()};np.savez(out/f'{arm}.npz',**states)
        report[arm]={key:base.metrics(states[key],q) for key,q in qs.items()}
        for key in ['old_train','old_valid']:report[arm][key]['drops']=core.drops(states[key],A[key])
        known[arm]=path;core.progress('evaluated',arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm,item in report.items():item['directional_ratios']=[item['new_valid'][x]['filtered_train_truths']['mrr']/report['none']['new_valid'][x]['filtered_train_truths']['mrr'] for x in core.DIRECTIONS]
    core.dump('evaluation/metrics.json',report);core.dump('evaluation/reused-equal-models.json',aliases)
    return report,time.monotonic()-tick

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial=torch.load(core.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load('/data/lc/untzh/risk58_20261013/cache/prefix10.pt',map_location='cpu',weights_only=True)
    z=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=z['old'],z['new']
    t=base.load(core.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz');d,probes,pc=prev.prepare(initial,new,t)
    rc=rank_probes(prefix,initial['R'],probes,core.truthmap(old,new));prev.probe_losses=rank_cost
    cal,cs=core.calibrate(initial,prefix,new,t,d);costs=dict(preparation=pc,rank_probe_preparation=rc,calibration_seconds=cs,training={})
    costs['training']['absolute_guard']=prev.train('absolute_guard','absolute',True,1.,initial,prefix,new,t,d,probes,cal)
    mean=costs['training']['absolute_guard']['mean_multiplier']
    costs['training']['absolute_meanmatched']=prev.train('absolute_meanmatched','absolute',False,mean,initial,prefix,new,t,d,probes,cal)
    core.dump('costs-running.json',costs)
    report,es=evaluate(old,new);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
