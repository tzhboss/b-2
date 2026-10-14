"""TRAIN-conditioned competition calibration without changing old scores."""
import collections
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,'/data/lc/untzh/search63_20261013')
import experiment63 as core
ex=core.ex
ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
BASES={
 'ordinary':ex.SRC/'matched_protection52_20261013/artifacts/train/new_only/epoch-20.pt',
 'uniform':core.PRIOR/'models/uniform/epoch-20.pt',
 'mixed':Path('/data/lc/untzh/balance68_20261014/models/mix50_none/epoch-20.pt'),
 'mixed_protected':Path('/data/lc/untzh/balance68_20261014/models/mix50_absolute/epoch-20.pt'),
}
ARMS={
 **{f'{base}_{kind}_{int(round(gamma*100))}':(base,kind,gamma)
     for base in BASES for kind in ['query_scale','relation_scale'] for gamma in [.7,.8,.9,.95]},
 **{f'{base}_query_shift_{int(round(delta*100))}':(base,'query_shift',delta)
     for base in BASES for delta in [.1,.25,.5]},
}
EXPOSED=set()
RELATION_KNOWN=None


@torch.no_grad()
def ranks(E,R,queries,truths,arm=None,split=None):
    _,kind,value=ARMS[arm]
    result=[]
    for lo in range(0,len(queries),64):
        a=queries[lo:lo+64];q=torch.as_tensor(a,device='cuda')
        scores=(E[q[:,0]]*R[q[:,1]])@E.T
        if kind in ['query_scale','query_shift']:
            # New-headed queries are never calibrated; preserve those learning directions.
            mask=torch.as_tensor([int(h)<ex.N and (int(h),int(r)) not in EXPOSED for h,r,t in a],device='cuda')
            if kind=='query_scale':
                scores[:,ex.N:]*=torch.where(mask,value,1.)[:,None]
            else:
                tau=scores[:,:ex.N].std(dim=1).clamp_min(1e-8)
                scores[:,ex.N:]-=(mask*tau*value)[:,None]
        elif kind=='relation_scale':
            eligible=(q[:,0]<ex.N)[:,None]&(~RELATION_KNOWN[q[:,1]])
            scores[:,ex.N:]*=torch.where(eligible,value,1.)
        else:raise ValueError(kind)
        gold=scores[torch.arange(len(q),device='cuda'),q[:,2]].clone()
        raw=1+(scores>gold[:,None]).sum(-1)
        rows=[];cols=[]
        for i,(h,r,t) in enumerate(a):
            for c in truths[int(h),int(r)]:
                if c!=int(t):rows.append(i);cols.append(c)
        if rows:scores[rows,cols]=-torch.inf
        filtered=1+(scores>gold[:,None]).sum(-1)
        result.append(torch.stack([raw,filtered],dim=1).cpu().numpy())
        if lo%4096==0:ex.dump('status.json',{'stage':'evaluate','arm':arm,'split':split,'completed':lo+len(a),'total':len(queries)})
    return np.concatenate(result)


def main():
    global EXPOSED,RELATION_KNOWN
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz')
    EXPOSED={(int(h),int(r)) for h,r,t in d['new'] if t>=ex.N}
    relation=np.zeros((474,ex.TOTAL-ex.N),bool)
    for h,r,t in d['new']:
        if t>=ex.N:relation[int(r),int(t)-ex.N]=True
    RELATION_KNOWN=torch.as_tensor(relation,device='cuda')
    sources={base:torch.load(path,map_location='cpu',weights_only=True) for base,path in BASES.items()}
    for name,(base,kind,value) in ARMS.items():
        out=ROOT/'models'/name;out.mkdir(parents=True)
        torch.save(sources[base],out/'epoch-20.pt')
    ex.dump('calibration-configs.json',{'arms':ARMS,'sources':{k:str(v) for k,v in BASES.items()},
                                     'TRAIN_exposed_queries':len(EXPOSED),'relation_known_candidates':int(relation.sum())})
    ex.base.ranks=ranks
    _,evaluation=core.evaluate(d['old'],d['new'],ARMS)
    costs={'evaluation':evaluation,'total_seconds':time.monotonic()-started,
           'training':'reused; source training costs required','selection':'Fixed exploratory grid; VALID exposed development'}
    ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
