import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
with np.load(ROOT/'queries.npz') as z:
    queries={k:z[k] for k in z.files}
report={}
for name,q in queries.items():
    uq=np.unique(q,axis=0)
    groups=np.unique(q[:,:2],axis=0)
    report[name]=dict(source_rows=len(q),unique_direction_facts=len(uq),duplicate_rows=len(q)-len(uq),query_groups=len(groups))
(ROOT/'source-counts.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
