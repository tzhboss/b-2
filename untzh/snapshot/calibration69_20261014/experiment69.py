"""Exploratory candidate-age score calibration; preserve all failures."""
import json
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0,'/data/lc/untzh/search63_20261013')
import experiment63 as core
ex=core.ex
ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
BASES={
 'contrastive':Path('/data/lc/untzh/probability67_20261014/models/contrastive_none/epoch-20.pt'),
 'mixed':Path('/data/lc/untzh/balance68_20261014/models/mix50_none/epoch-20.pt'),
 'mixed_protected':Path('/data/lc/untzh/balance68_20261014/models/mix50_absolute/epoch-20.pt'),
 'ordinary':ex.SRC/'matched_protection52_20261013/artifacts/train/new_only/epoch-20.pt',
 'uniform':core.PRIOR/'models/uniform/epoch-20.pt',
}
ARMS={f'{base}_g{int(round(gamma*100))}':(base,gamma)
      for base in BASES for gamma in ([.7,.8,.85,.9,.95,1.] if base not in ['ordinary','uniform'] else [.9,.95])}


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    materialization={}
    for name,(base,gamma) in ARMS.items():
        m=torch.load(BASES[base],map_location='cpu',weights_only=True)
        if not torch.isfinite(m['E']).all():raise FloatingPointError(base)
        before=m['E'][:ex.N].clone();R=m['R'].clone()
        m['E'][ex.N:]*=gamma
        assert torch.equal(m['E'][:ex.N],before) and torch.equal(m['R'],R)
        out=ROOT/'models'/name;out.mkdir(parents=True)
        torch.save(m,out/'epoch-20.pt')
        materialization[name]={'source':str(BASES[base]),'gamma':gamma,'new_entity_norm_mean':float(m['E'][ex.N:].norm(dim=1).mean())}
    ex.dump('calibration-models.json',materialization)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz')
    _,evaluation=core.evaluate(d['old'],d['new'],ARMS)
    costs={'evaluation':evaluation,'total_seconds':time.monotonic()-started,
           'training':'No retraining; reused 052/058/067/068 checkpoints. Historical learning costs remain necessary.',
           'selection':'Fixed exploratory gamma grid; exposed VALID is development evidence, not independent confirmation.'}
    ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
