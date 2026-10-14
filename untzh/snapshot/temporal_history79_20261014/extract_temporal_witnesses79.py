"""Extract cached exact prediction witnesses for dated organization histories."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lc/untzh/temporal_drift76_20261014')
mapping = json.loads((SOURCE/'prepared/mapping.json').read_text())
data = np.load(SOURCE/'prepared/data.npz')
names = ['Aam Aadmi Party', 'National United Front for Democracy Against Dictatorship',
         'Liberal National Party', 'Party President (Liberal National Party)', 'Your Party']
nr = len(mapping['relations'])
out = {}
for file in sorted((ROOT/'icews').glob('*-history.npz')):
    hist = np.load(file)
    cases = []
    for i in np.flatnonzero(np.isin(hist['prediction_after'],
                                  [mapping['entities'].index(n) for n in names])):
        idx = int(hist['ids'][i])
        h,r,t,d = map(int,data['test'][idx])
        cases.append({'test_directional_index':idx,
                      'fixed_date':mapping['dates'][d], 'direction':'tail' if r<nr else 'head',
                      'query_entity':mapping['entities'][h], 'relation':mapping['relations'][r%nr],
                      'gold':mapping['entities'][t],
                      'before_top1':mapping['entities'][int(hist['prediction_before'][i])],
                      'after_top1':mapping['entities'][int(hist['prediction_after'][i])],
                      'before_true':bool(hist['before_top1_recorded_true'][i]),
                      'after_true':bool(hist['after_top1_recorded_true'][i]),
                      'before_rank':float(hist['original_rank'][i]),
                      'after_rank':float(hist['final_rank'][i]),
                      'rank_worsening':bool(hist['final_rank'][i]>hist['original_rank'][i])})
    out[file.name.removesuffix('-history.npz')] = cases
(ROOT/'temporal-witness-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out['TComplEx_finetune'],ensure_ascii=False,indent=2))
