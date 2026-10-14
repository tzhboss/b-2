"""Read existing scores to bound false competitors; no model execution."""
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lc/untzh/temporal_drift76_20261014')
t0 = time.perf_counter()
sample = json.loads((ROOT/'representative-history-sample.json').read_text())
evidence = json.loads((ROOT/'representative-history-evidence.json').read_text())
temporal = json.loads((ROOT/'temporal-competitor-evidence.json').read_text())
mapping = json.loads((SOURCE/'prepared/mapping.json').read_text())
with np.load(SOURCE/'prepared/data.npz') as z:
    test = z['test']
    filter_rows, filter_cols = z['filter_rows'], z['filter_cols']
with np.load(ROOT/'fixed40-score-cache.npz') as z:
    ids, before, after, ns = [z[k] for k in ['ids','before','after','origin_active_n']]
records = []
for i, s in enumerate(sample['records']):
    idx = s['test_directional_index']
    assert idx == ids[i]
    g = int(test[idx,2]); n = int(ns[i])
    b = before[i,:n] - before[i,g]
    a = after[i] - after[i,g]
    b_ids = np.flatnonzero((b>0)&(np.arange(n)!=g))
    b_ties = np.flatnonzero((b==0)&(np.arange(n)!=g))
    lo, hi = np.searchsorted(filter_rows,[idx,idx+1])
    truth_ids = set(map(int,filter_cols[lo:hi]))
    bf_ids = [int(k) for k in b_ids if k not in truth_ids]
    bf_ties = [int(k) for k in b_ties if k not in truth_ids]
    assert 1+len(bf_ids)+.5*len(bf_ties)==s['original_gold_rank']
    witnesses = [w for w in temporal['witnesses'] if w['sample_id']==s['sample_id']]
    witness_ids = [w['entity_id'] for w in witnesses]
    assert len(witness_ids)==len(set(witness_ids))
    assert all(k not in truth_ids and a[k]>0 for k in witness_ids)
    known_false_after = len(witness_ids)
    raw_upper = len(b_ids)+.5*len(b_ties)
    filtered_upper = len(bf_ids)+.5*len(bf_ties)
    gold_supported = evidence['judgments'][s['sample_id']]['gold_truth']=='supported_at_query_semantic_granularity'
    raw_lower = known_false_after-raw_upper
    filtered_lower = known_false_after-filtered_upper
    records.append({
        'sample_id':s['sample_id'], 'gold_fact':s['gold_fact'],
        'gold_supported_at_query_semantic_granularity':gold_supported,
        'known_false_strictly_above_gold_after_n':known_false_after,
        'raw_before_above_gold_n':len(b_ids), 'raw_before_gold_other_ties_n':len(b_ties),
        'filtered_before_above_gold_n':len(bf_ids), 'filtered_before_gold_other_ties_n':len(bf_ties),
        'raw_before_possible_false_rank_contribution_upper':raw_upper,
        'world_false_rank_drift_lower_bound_without_benchmark_filter_soundness':raw_lower,
        'world_false_rank_drift_lower_bound_if_benchmark_filters_sound':filtered_lower,
        'guaranteed_error_drift_without_filter_soundness':bool(gold_supported and raw_lower>0),
        'guaranteed_error_drift_conditional_on_filter_soundness':bool(gold_supported and filtered_lower>0),
        'known_false_candidate_names':[w['candidate_name'] for w in witnesses],
        'raw_before_outranker_names':[mapping['entities'][int(k)] for k in b_ids],
        'raw_before_filtered_out_outranker_names':[mapping['entities'][int(k)] for k in b_ids if k in truth_ids],
    })
raw_confirmed = [r['sample_id'] for r in records if r['guaranteed_error_drift_without_filter_soundness']]
filter_confirmed = [r['sample_id'] for r in records if r['guaranteed_error_drift_conditional_on_filter_soundness']]
out = {
    'sample_n':40, 'population_worsened_query_n':sample['population_n'],
    'definition':'Reality-corrected rank counts false query answers above a valid gold, with half weight for ties; hypothetical true alternatives are filtered at both checkpoints.',
    'inequality':'D_false >= known_false_above_after - all_possibly_false_above_before. Before upper can use raw scores without assuming benchmark positives are sound; filtered upper requires their soundness.',
    'conditions':['independent gold evidence accepted at query semantic granularity','temporal witnesses refer to the named organization or dated affiliation','existing cached scores and candidate spaces define the compared ranks'],
    'raw_confirmed_query_ids':raw_confirmed,
    'raw_confirmed_query_n':len(raw_confirmed),
    'raw_fixed_sample_error_drift_lower_bound':len(raw_confirmed)/40,
    'benchmark_filter_soundness_conditional_query_ids':filter_confirmed,
    'benchmark_filter_soundness_conditional_query_n':len(filter_confirmed),
    'benchmark_filter_soundness_conditional_fixed_sample_lower_bound':len(filter_confirmed)/40,
    'world_error_population_fraction':None, 'world_reasonable_population_fraction':None,
    'not_population_estimate_or_complete_UFR':True,
    'training':False,'neural_rescoring':False,
    'records':records, 'seconds':time.perf_counter()-t0,
}
(ROOT/'false-rank-drift-bounds-round10.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},ensure_ascii=False))
for r in records:
    if r['known_false_strictly_above_gold_after_n']:
        print(json.dumps(r,ensure_ascii=False))
