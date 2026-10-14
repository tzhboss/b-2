import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
summary_path=ROOT/'summary.json'
summary=json.loads(summary_path.read_text())
backup=ROOT/'summary-before-metadata-fix.json'
assert not backup.exists()
backup.write_text(summary_path.read_text())
records=[]
for seed,report in summary['seeds'].items():
    out=ROOT/f'seed{seed}'
    data=np.load(out/'gates-frozen.npz',allow_pickle=False)
    used=report['costs']['training']['forecast_anchor']['steps']
    fg=data['forecast_anchor'][:used]
    rg=data['random_matched'][:used]
    cg=data['current_anchor'][:used]
    assert np.array_equal(fg.sum(1),rg.sum(1))
    for arm,g in [('forecast_anchor',fg),('random_matched',rg),('current_anchor',cg)]:
        steps=json.loads((out/'models'/arm/'steps.json').read_text())
        assert [r['selected_queries'] for r in steps]==g.sum(1).tolist()
    prep=report['costs']['preparation']
    prep['original_cached_selection_rows']=len(data['forecast_anchor'])
    prep['used_selection_rows']=used
    prep['foreground_draws']=int(fg.sum())
    prep['all_query_draws']=int(fg.size)
    prep['forecast_trigger_steps']=int(fg.any(1).sum())
    prep['current_trigger_steps']=int(cg.any(1).sum())
    prep['post_completion_metadata_correction']=True
    (out/'preparation-before-metadata-fix.json').write_text((out/'preparation.json').read_text())
    (out/'preparation.json').write_text(json.dumps(prep,ensure_ascii=False,indent=2))
    records.append(dict(seed=seed,used_steps=used,query_draws=int(fg.sum()),query_draw_total=int(fg.size),
                        foreground_random_step_counts_equal=True,committed_steps_match=True))
summary['metadata_correction']='preparation counted all 370 cached selection rows; training used original first220. Corrected reporting only; no training or evaluation repeated.'
summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(ROOT/'metadata-correction.json').write_text(json.dumps(dict(no_training_or_scoring=True,records=records),indent=2))
print(json.dumps(records))
