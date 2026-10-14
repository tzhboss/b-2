import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/budget_action90_20261014')
import experiment90 as action
route,core,base=action.route,action.core,action.base
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/increment87_20261014')
action.ROOT=route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=ROOT
route.SOURCE=SOURCE
SPECS={f'{s}_uniform':(s,'uniform') for s in ['absolute128','absolute_full','violation_full','budget_full']}
route.ARMS=list(SPECS)

def prepare():
    start=time.monotonic()
    initial=torch.load(SOURCE/'cache/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(SOURCE/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    t=base.load(SOURCE/'cache/learning.npz');z=base.load(SOURCE/'cache/train.npz')
    old,new=z['old'],z['new'];truths=core.truthmap(old,np.concatenate([new,z['extra']]))
    c=base.load(SOURCE/'cache/cohort.npz');teacher=base.load(SOURCE/'cache/teacher-train.npz')['old_train']
    ranks=teacher[c['ids'],1];assert (ranks<=10).all() and len(t['epochs'])==440
    q=torch.as_tensor(c['q'],device='cuda');e=initial['E'].cuda().double()
    gold=(q*e[c['queries'][:,2]]).sum(-1)
    fixed=np.zeros((len(ranks),8),np.float32);fixed[:,0]=ranks
    d=dict(q=q,candidates=torch.as_tensor(c['candidates'],device='cuda'),gold=gold,fixed=fixed)
    assert torch.equal(prefix['E'][:action.N],initial['E'][:action.N])
    seconds=action.extra_data(truths,d)
    return initial,prefix,new,t,truths,d,time.monotonic()-start

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,new,t,truths,d,ps=prepare()
    costs=dict(preparation_seconds=ps,source=str(SOURCE),seed=17,normalization='rho .1 on each nonzero protection gradient; identical to090',training={})
    for name,spec in SPECS.items():
        costs['training'][name]=action.train(name,spec,initial,prefix,new,t,d,None)
        core.dump('costs-running.json',costs)
    metrics,es=route.evaluate(truths);costs.update(evaluation_seconds=es,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
