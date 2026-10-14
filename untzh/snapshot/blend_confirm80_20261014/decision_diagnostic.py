"""Read cached TRAIN trajectory to assess global onset signals; no new scoring."""
import json
from pathlib import Path
import numpy as np
source=Path('/data/lc/untzh/risk58_20261013/cache')
root=Path(__file__).resolve().parent
cohort=np.load(source/'cohort.npz')
rows=[]
for ep in range(20):
    with np.load(source/f'frame-{ep:02d}.npz') as f: X=f['X'];r=f['ranks']
    with np.load(source/f'frame-{ep+1:02d}.npz') as f: nr=f['ranks']
    safe=r<=10;event=safe&(nr>10)
    rows.append(dict(epoch=ep,safe=int(safe.sum()),new_drop_events=int(event.sum()),
        dropped_fraction=float((r>10).mean()),
        pressure32_positive_fraction=float((X[:,8]>0).mean()),
        pressure128_positive_fraction=float((X[:,9]>0).mean()),
        pressure32_near_fraction=float((X[:,8]>-.25).mean()),
        pressure32_quantiles=np.quantile(X[:,8],[.5,.9,.99]).tolist(),
        mean_mass_proxy=float(X[:,14].mean())))
report=dict(source=str(source),scope='already exposed seed17 TRAIN-only cached ordinary trajectory; no independent confirmation',
    rows=rows,warning='new-candidate pressure is not equal to dropout or benefit; global onset thresholds below are diagnostic only')
(root/'decision-diagnostic.json').write_text(json.dumps(report,indent=2))
for r in rows:print(r)
