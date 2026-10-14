"""Check what raw Top1 evidence can explain about cached filtered ranks."""
import json
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/temporal_drift76_20261014')
started=time.perf_counter()
sample=json.loads((ROOT/'representative-history-sample.json').read_text(encoding='utf-8'))
with np.load(ROOT/'icews/TComplEx_finetune-history.npz') as z:
 h={k:z[k] for k in z.files}
with np.load(SOURCE/'evaluation/TComplEx_finetune-2015.npz') as z:
 final={k:z[k] for k in ('ids','gold_score','filtered_full')}
records=[]
for c in sample['records']:
 i=int(np.searchsorted(h['ids'],c['test_directional_index']))
 j=int(np.searchsorted(final['ids'],c['test_directional_index']))
 assert h['ids'][i]==final['ids'][j]==c['test_directional_index']
 assert float(final['filtered_full'][j])==c['final_gold_rank']
 # Only subtract for externally supported alternatives that were not already
 # filtered. This is a labelled diagnostic; do not rewrite benchmark ranks.
 top_score=float(h['top1_after_score'][i]);gold_score=float(final['gold_score'][j])
 d={'sample_id':c['sample_id'],'final_top1_already_known_and_filtered':c['after_recorded'],
    'top1_score':top_score,'gold_score':gold_score,'top1_minus_gold_score':top_score-gold_score,
    'raw_top1_ties':int(h['after_top1_ties'][i]),'benchmark_filtered_gold_rank':c['final_gold_rank'],
    'semantic_alternative_filter_diagnostic':None}
 if c['sample_id']=='S39':
  assert not c['after_recorded'] and c['final_gold_rank']==2 and c['original_gold_rank']==1
  assert top_score>gold_score and d['raw_top1_ties']==1
  d['semantic_alternative_filter_diagnostic']={
    'conditional_filter_rank_before':1.0,'conditional_filter_rank_after':1.0,
    'benchmark_drift':1.0,'conditional_drift_after_removing_supported_iraq':0.0,
    'assumptions':'Public article Iraq/Yemen Consult mapping accepted at country-query granularity; original China gold retained as benchmark target, not externally validated on exact date; cached score comparisons compatible across batches.',
    'interpretation':'One omitted semantically supported alternative suffices to remove this observed filtered-rank worsening. Does not prove China event date, causal beneficial model update, or reality truth of all rankings.'}
 if c['sample_id']=='S33':
  assert c['after_recorded']
  d['interpretation']='Xi is already a recorded positive and removed from filtered Qin rank. His real-world speech cannot explain Qin filtered rank3->62. Raw Top1 truth is not a classification of filtered-rank drift.'
 records.append(d)
out={'scope':'Cached arrays only; no training or neural rescoring; original benchmark unchanged.',
     'records':records,'supported_missing_alternative_removes_drift_n':1,
     'fixed_sample_fraction_with_this_verified_conditional_filter_explanation':1/40,
     'not_a_reasonable_or_error_rate':True,'seconds':time.perf_counter()-started}
(ROOT/'sample-filter-interpretation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({**{k:v for k,v in out.items() if k!='records'},'illustrations':[r for r in records if r['sample_id'] in {'S33','S39'}]},ensure_ascii=False,indent=2))
