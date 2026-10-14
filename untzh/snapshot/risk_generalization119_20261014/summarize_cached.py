import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
cx=Path('/data/lc/untzh/complex_transfer118_20261014')
stages={}
for mode in ['standard','low']:
    for split in ['old_train','old_valid']:
        with np.load(cx/'evaluation'/f'{mode}-{split}-phases.npz') as z:
            states={k:z[k][:,1] for k in z.files}
        eligible=states['old_candidates']<=10
        drops={k:eligible&(a>10) for k,a in states.items()}
        transitions={}
        for before,after in [('old_candidates','expanded_initial'),('expanded_initial','epoch10'),('epoch10','epoch20')]:
            transitions[f'{before}_to_{after}']=dict(new_drops=int((eligible&~drops[before]&drops[after]).sum()),rescued=int((drops[before]&~drops[after]).sum()))
        stages[f'{mode}-{split}']=dict(eligible=int(eligible.sum()),drops={k:int(v.sum()) for k,v in drops.items()},transitions=transitions)
out=cx/'phase-summary.json';out.write_text(json.dumps(stages,indent=2),encoding='utf8')
print('COMPLEX_PHASE',json.dumps(stages))
s=json.loads((root/'summary.json').read_text(encoding='utf8'));flat={}
for k,v in s['cases'].items():
    m={n:dict(AP=x['safe_ap'],safe5=x['policies']['current_safe']['0.05'],all5=x['policies']['all_reference']['0.05']) for n,x in v['methods'].items()}
    flat[k]=dict(n=v['n'],safe=v['safe_at_prefix'],events=v['new_events'],rescued=v['rescued'],methods=m,
                 features_seconds=v['features']['seconds'],prediction_seconds=v['predict_seconds'])
(root/'summary-flat.json').write_text(json.dumps(flat,indent=2),encoding='utf8')
print('RISK_FLAT',json.dumps(flat))
