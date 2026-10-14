import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/anchor_rank103_20261014')
import experiment103 as previous
core,base=previous.core,previous.base
ROOT=Path(__file__).resolve().parent
N=11258

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

class FastOld:
    def __init__(self,seed,truths):
        self.seed=seed;self.truths=truths
        source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        full=Path('/data/lc/untzh/full_valid102_20261014')
        initial=torch.load(source/'cache/initial.pt',map_location='cpu',weights_only=True)
        self.oldE=initial['E'][:N].double().cuda();self.R=initial['R'].double().cuda()
        self.queries={'old_train':base.load(source/'evaluation/queries.npz')['old_train'],
                      'old_valid':base.load(full/'queries.npz')['old_valid']}
        self.teacher={'old_train':base.load(source/'evaluation/teacher.npz')['old_train'],
                      'old_valid':base.load(full/f'seed{seed}/teacher.npz')['old_valid']}
        self.cache={};tick=time.monotonic();costs={}
        for split,a in self.queries.items():
            start=time.monotonic();path=ROOT/f'seed{seed}'/f'{split}.npz'
            if path.exists():z=base.load(path);reused=True
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                gold=[];rows=[];cols=[]
                for i in range(0,len(a),64):
                    b=a[i:i+64];q=torch.as_tensor(b,device='cuda')
                    scores=(self.oldE[q[:,0]]*self.R[q[:,1]])@self.oldE.T
                    gold.append(scores[torch.arange(len(b),device='cuda'),q[:,2]].cpu().numpy())
                    for j,(h,r,g) in enumerate(b):
                        for c in truths[int(h),int(r)]:
                            if c>=N:rows.append(i+j);cols.append(c-N)
                z=dict(gold=np.concatenate(gold),rows=np.asarray(rows,dtype=np.int64),cols=np.asarray(cols,dtype=np.int64))
                np.savez(path,**z);reused=False
            self.cache[split]=dict(gold=torch.as_tensor(z['gold'],device='cuda'),rows=z['rows'],cols=z['cols'])
            costs[split]=dict(seconds=time.monotonic()-start,reused=reused,bytes=sum(v.nbytes for v in z.values()))
        self.preparation=dict(seconds=time.monotonic()-tick,splits=costs,historical_teacher_not_free=True)

    def ranks(self,E,R,split):
        assert torch.equal(E[:N],self.oldE) and torch.equal(R,self.R)
        tick=time.monotonic();a=self.queries[split];cache=self.cache[split];parts=[];fallback=0
        for i in range(0,len(a),64):
            b=a[i:i+64];q=torch.as_tensor(b,device='cuda');gold=cache['gold'][i:i+len(b)]
            scores=(E[q[:,0]]*R[q[:,1]])@E[N:].T
            close=(scores-gold[:,None]).abs()<=1e-10*(1+gold.abs()[:,None])
            if close.any():
                parts.append(base.ranks(E,R,b,self.truths));fallback+=len(b);continue
            raw=(scores>gold[:,None]).sum(1).cpu().numpy()
            lo=np.searchsorted(cache['rows'],i);hi=np.searchsorted(cache['rows'],i+len(b))
            if hi>lo:scores[cache['rows'][lo:hi]-i,cache['cols'][lo:hi]]=-torch.inf
            filtered=(scores>gold[:,None]).sum(1).cpu().numpy()
            parts.append(self.teacher[split][i:i+len(b)]+np.column_stack([raw,filtered]))
        result=np.concatenate(parts)
        return result,dict(seconds=time.monotonic()-tick,fallback_queries=fallback,n=len(a),new_candidates=1876)

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base.ROOT=ROOT;tick=time.monotonic();status('starting')
    z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]))
    reports={}
    for seed in [29,17]:
        status('prepare',seed=seed);fast=FastOld(seed,truths);cases={}
        source=Path(f'/data/lc/untzh/full_replay105_20261014/seed{seed}')
        for arm in ['full_replay_joint','full_replay_anchor']:
            status('validate',seed=seed,arm=arm)
            m=torch.load(source/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
            saved=base.load(source/'evaluation'/f'{arm}.npz');cases[arm]={}
            for split in ['old_train','old_valid']:
                state,cost=fast.ranks(m['E'].cuda(),m['R'].cuda(),split)
                equal=np.array_equal(state,saved[split])
                cases[arm][split]=dict(equal=equal,mismatch_rows=int(np.any(state!=saved[split],axis=1).sum()),max_rank_difference=int(np.abs(state-saved[split]).max()),**cost)
                print('FAST107',seed,arm,split,cases[arm][split],flush=True)
        reports[str(seed)]=dict(preparation=fast.preparation,cases=cases)
    equal=all(c['equal'] for s in reports.values() for a in s['cases'].values() for c in a.values())
    base.write(ROOT/'summary.json',dict(seeds=reports,all_equal=equal,total_seconds=time.monotonic()-tick))
    status('complete' if equal else 'needs_original_evaluation',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
