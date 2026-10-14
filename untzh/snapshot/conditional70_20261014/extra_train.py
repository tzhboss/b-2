"""Full TRAIN evaluation of two near-frontier candidates; no retraining."""
import collections
import json
import time
import numpy as np
import torch
import experiment70 as run

torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
started=time.monotonic()
d=run.ex.base.load(run.ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz')
truth=collections.defaultdict(set)
for h,r,t in np.concatenate([d['old'],d['new']]):truth[int(h),int(r)].add(int(t))
run.EXPOSED={(int(h),int(r)) for h,r,t in d['new'] if t>=run.ex.N}
relation=np.zeros((474,run.ex.TOTAL-run.ex.N),bool)
for h,r,t in d['new']:
    if t>=run.ex.N:relation[int(r),int(t)-run.ex.N]=True
run.RELATION_KNOWN=torch.as_tensor(relation,device='cuda')
A=run.ex.base.load(run.ex.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
report=json.loads((run.ROOT/'evaluation/metrics.json').read_text())
for name in ['mixed_protected_query_scale_90','mixed_protected_query_shift_10']:
    m=torch.load(run.ROOT/'models'/name/'epoch-20.pt',map_location='cpu',weights_only=True)
    a=run.ranks(m['E'].cuda(),m['R'].cuda(),d['old'],truth,arm=name,split='old_train')
    np.savez(run.ROOT/'evaluation'/f'{name}-train.npz',old_train=a)
    report[name]['old_train']=run.ex.base.metrics(a,d['old'])
    report[name]['old_train']['drops']=run.core.dropstats(a,A['old_train'])
    print(name,report[name]['old_train']['drops'],flush=True)
run.ex.dump('evaluation/metrics.json',report)
valid=run.ex.base.load(run.ex.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
mask=valid['new'][:,0]>=run.ex.N
paths={'ordinary':run.ex.SRC/'replay_strength53_20261013/artifacts/evaluation/new_only-e20.npz',
       'uniform':run.core.PRIOR/'evaluation/uniform.npz',
       'mixed':run.Path('/data/lc/untzh/balance68_20261014/evaluation/mix50_none-valid.npz'),
       'mixed_protected':run.Path('/data/lc/untzh/balance68_20261014/evaluation/mix50_absolute-valid.npz')}
baseline={base:run.ex.base.load(path)['new_valid'][mask] for base,path in paths.items()}
for name,(base,kind,value) in run.ARMS.items():
    a=run.ex.base.load(run.ROOT/'evaluation'/f'{name}-valid.npz')['new_valid'][mask]
    assert np.array_equal(a,baseline[base]),name
run.ex.dump('supplement.json',{'seconds':time.monotonic()-started,'extra_train_arms':['mixed_protected_query_scale_90','mixed_protected_query_shift_10'],
                              'new_headed_ranks_unchanged_all_44':True})
run.ex.progress('complete',supplement_seconds=time.monotonic()-started)
