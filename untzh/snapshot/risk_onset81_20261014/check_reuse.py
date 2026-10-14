"""Basic equality for ranking reuse; no scoring or independent audit."""
import json
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
out={}
for seed in [29,43]:
    candidate=torch.load(root/f'seed{seed}/models/risk010/epoch-20.pt',map_location='cpu',weights_only=True)
    control=torch.load(Path('/data/lc/untzh/blend_confirm80_20261014')/f'seed{seed}/models/none/epoch-20.pt',map_location='cpu',weights_only=True)
    out[str(seed)]={k:torch.equal(candidate[k],control[k]) for k in ['E','R']}
    assert all(out[str(seed)].values())
(root/'basic-reuse-check.json').write_text(json.dumps(out,indent=2))
print(out)
