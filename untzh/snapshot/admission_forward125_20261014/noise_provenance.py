import json
import time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
tick=time.monotonic();rows=[]
for seed in [20261014,20261015,20261016]:
    f=torch.load(root/'cache'/f'43-noise-{seed}.pt',map_location='cpu',weights_only=True)
    w=torch.load(root/'cache'/f'WN_DistMult-noise-{seed}.pt',map_location='cpu',weights_only=True)
    rows.append(dict(seed=seed,fb_same_shape_equal={str(p):bool(torch.equal(f,torch.load(root/'cache'/f'{p}-noise-{seed}.pt',map_location='cpu',weights_only=True))) for p in [17,29]},
                     wn_same_shape_equal=bool(torch.equal(w,torch.load(root/'cache'/f'WN_ComplEx-noise-{seed}.pt',map_location='cpu',weights_only=True))),
                     fb_equals_wn_prefix=bool(torch.equal(f,w[:len(f)]))))
report=dict(rows=rows,seconds=time.monotonic()-tick,scope='noise cache identity for dependence disclosure, no model/rank audit',
            rng_seed_settings=3,parent_seed_payloads=15,evaluation_units=45,parents_do_not_supply_independent_noise_replications=True)
(root/'realization-provenance.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
