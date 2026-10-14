"""Summarize cached prediction names for historical source lookup; no scoring."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lc/untzh/temporal_drift76_20261014')
mapping = json.loads((SOURCE/'prepared/mapping.json').read_text())
data = np.load(SOURCE/'prepared/data.npz')
history = np.load(ROOT/'icews/TComplEx_finetune-history.npz')
ids = history['ids']
worse = history['final_rank'] > history['original_rank']
new_error = worse & history['before_top1_recorded_true'] & ~history['after_top1_recorded_true']
nr = len(mapping['relations'])
records = []
for entity, n in Counter(history['prediction_after'][new_error].tolist()).most_common():
    idxs = np.flatnonzero(new_error & (history['prediction_after'] == entity))
    records.append({'entity_id':entity, 'name':mapping['entities'][entity],
                    'n':n, 'train_first_year':int(data['eyear'][entity]),
                    'actor_n':int((data['test'][ids[idxs],1]>=nr).sum())})
(ROOT/'prediction-entities.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps([x for x in records if x['train_first_year']>2005],ensure_ascii=False,indent=2))
