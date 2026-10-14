import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/wn_transfer112_20261014')
import experiment112 as source
base=source.base
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/wn_transfer112_20261014')

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    base.ROOT=ROOT;tick=time.monotonic();status('prepare')
    initial=torch.load(SOURCE/'cache/initial.pt',map_location='cpu',weights_only=True);prefix=torch.load(SOURCE/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    train=base.load(SOURCE/'cache/train.npz');truths=source.core.truthmap(train['old'],train['new']);queries=base.load(SOURCE/'evaluation/queries.npz');final=base.load(SOURCE/'evaluation/none.npz');reports={}
    for split in ['old_train','old_valid']:
        status('score',split=split);z=base.load(SOURCE/'cache'/f'fast-{split}.npz');q=queries[split]
        expanded,cost0=source.fast_ranks(initial['E'].cuda().double(),initial['R'].cuda().double(),q,z,truths)
        halfway,cost10=source.fast_ranks(prefix['E'].cuda().double(),initial['R'].cuda().double(),q,z,truths)
        states={'old_candidates':z['teacher'],'expanded_initial':expanded,'epoch10':halfway,'epoch20':final[split]}
        result={}
        for col,name in enumerate(['raw','filtered_train_truths']):
            eligible=z['teacher'][:,col]<=10;drops={phase:(a[:,col]>10)&eligible for phase,a in states.items()}
            phases={phase:dict(drops=int(mask.sum()),eligible=int(eligible.sum()),mrr=float((1/states[phase][:,col]).mean())) for phase,mask in drops.items()}
            transitions={}
            for before,after in [('old_candidates','expanded_initial'),('expanded_initial','epoch10'),('epoch10','epoch20')]:
                transitions[f'{before}_to_{after}']=dict(new_drops=int((~drops[before]&drops[after]&eligible).sum()),rescued=int((drops[before]&~drops[after]&eligible).sum()),persistent_drops=int((drops[before]&drops[after]).sum()))
            result[name]=dict(phases=phases,transitions=transitions)
        reports[split]=dict(metrics=result,scoring=dict(expanded=cost0,epoch10=cost10));np.savez(ROOT/f'{split}-phases.npz',**states)
        print('PHASE114',split,result['filtered_train_truths'],flush=True)
    base.write(ROOT/'summary.json',dict(splits=reports,no_training=True,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
