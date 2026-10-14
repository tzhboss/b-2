"""Supplement results from saved ranks/receipts only; no model recomputation."""
import json
import numpy as np
from pathlib import Path
import base_experiment as b

root = Path(__file__).resolve().parent
costs = json.loads((root/'costs.json').read_text())
paths = [b.SRC/'matched_protection52_20261013/artifacts/tape/receipt.json',
         b.SRC/'matched_protection52_20261013/artifacts/train/new_only/receipt.json',
         b.SRC/'matched_protection52_20261013/artifacts/train/absolute_replay/receipt.json',
         b.SRC/'replay_strength53_20261013/artifacts/train/replay_010/receipt.json',
         b.SRC/'replay_strength53_20261013/artifacts/train/replay_030/receipt.json',
         Path('/data/lc/untzh/signals56_20261013/costs.json')]
costs['historical_cost_records'] = {str(p):json.loads(p.read_text()) for p in paths}
costs['paired_addon_training_ratio'] = costs['train']['replay_plus_meor_stepcap']['seconds']/costs['train']['replay']['seconds']
costs['paired_addon_training_ratio_note'] = 'Training only, asynchronous timing, excludes initial calibration and historical preparation; not deployment ratio.'
b.write(root/'costs.json',costs)
fresh = b.load(root/'evaluation/replay.npz')
historical = b.load(b.SRC/'replay_strength53_20261013/artifacts/evaluation/absolute_replay-e20.npz')
equal = {k:bool(np.array_equal(v,historical[k])) for k,v in fresh.items()}
if not all(equal.values()):
    raise ValueError('Fresh replay ranks differ from historical same-contract ranks')
b.write(root/'replay_reproduction.json',{'saved_rank_arrays_equal':equal,
         'scope':'Basic endpoint consistency, no additional scoring or independent audit.',
         'historical_rank_source':str(b.SRC/'replay_strength53_20261013/artifacts/evaluation/absolute_replay-e20.npz')})
print('REPLAY_SAVED_RANKS_EQUAL',equal)
