"""Directional near-Pareto candidate: supplementary TRAIN evaluation only."""
import collections
import json
import torch
import numpy as np
import experiment64 as run

torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
d=run.ex.base.load(run.ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz')
truth=collections.defaultdict(set)
for h,r,t in np.concatenate([d['old'],d['new']]):truth[int(h),int(r)].add(int(t))
A=run.ex.base.load(run.ex.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
report=json.loads((run.ROOT/'evaluation/metrics.json').read_text())
with torch.no_grad():
    name='query_old_only'
    m=torch.load(run.ROOT/'models'/name/'epoch-20.pt',map_location='cpu',weights_only=True)
    a=run.ex.base.ranks(m['E'].cuda(),m['R'].cuda(),d['old'],truth,arm=name,split='old_train')
    np.savez(run.ROOT/'evaluation'/f'{name}-train.npz',old_train=a)
    report[name]['old_train']=run.ex.base.metrics(a,d['old'])
    report[name]['old_train']['drops']=run.core.dropstats(a,A['old_train'])
    run.ex.dump('evaluation/metrics.json',report)
    print(name,report[name]['old_train']['drops'],flush=True)
