"""Tight candidate-wise fixed-truth bounds from existing scores; no model evaluation."""
import json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/temporal_drift76_20261014')
target=ROOT/'paired-truth-drift-bounds-round19.json'
if target.exists():
    print('EXISTING_OUTPUT_NO_RECOMPUTATION')
    raise SystemExit(0)
start=time.perf_counter()
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
sample=read('representative-history-sample.json')['records']
evidence=read('representative-history-evidence.json')
org=read('temporal-competitor-evidence.json')['witnesses']
life=read('lifetime-crossing-interpretation-round12.json')['records']+read('pope-crossing-interpretation-round18.json')['records']
mapping=json.loads((SOURCE/'prepared/mapping.json').read_text())
names=mapping['entities'];lookup={n:i for i,n in enumerate(names)}
with np.load(ROOT/'fixed40-score-cache.npz') as z:cache={k:z[k] for k in z.files}
with np.load(SOURCE/'prepared/data.npz') as z:data={k:z[k] for k in ['test','filter_rows','filter_cols']}
assert len(sample)==40 and cache['after'].shape[0]==40
assert np.array_equal(cache['ids'],[r['test_directional_index'] for r in sample])
boundaries=np.searchsorted(data['filter_rows'],np.arange(len(data['test'])+1))
# Only externally supported candidates; do not grant all benchmark positives world truth.
supported={'S33':{lookup['Xi Jinping']},'S39':{lookup['Iraq']}}
records=[]
for i,r in enumerate(sample):
    sid=r['sample_id'];idx=int(cache['ids'][i]);gold=int(data['test'][idx,2]);n=int(cache['origin_active_n'][i])
    a=cache['after'][i];b=cache['before'][i]
    wa=(a>a[gold]).astype(float)+.5*(a==a[gold]);wb=np.zeros(len(a))
    wb[:n]=(b[:n]>b[gold]).astype(float)+.5*(b[:n]==b[gold])
    wa[gold]=wb[gold]=0;delta=wa-wb
    truths=set(map(int,data['filter_cols'][boundaries[idx]:boundaries[idx+1]]))
    filt=np.ones(len(a),dtype=bool);filt[list(truths)]=False;filt[gold]=False
    assert 1+float(wb[filt].sum())==r['original_gold_rank']
    assert 1+float(wa[filt].sum())==r['final_gold_rank']
    false_org={int(w['entity_id']) for w in org if w['sample_id']==sid}
    false_life={int(w['entity_id']) for w in life if w['sample_id']==sid}
    gold_ok=evidence['judgments'][sid]['gold_truth']=='supported_at_query_semantic_granularity'
    def bound(false_ids,true_ids):
        assert not false_ids & true_ids
        assert gold not in false_ids
        unknown=np.ones(len(a),dtype=bool);unknown[gold]=False
        unknown[list(false_ids|true_ids)]=False
        known=float(delta[list(false_ids)].sum()) if false_ids else 0.0
        low=known+float(np.minimum(delta[unknown],0).sum())
        high=known+float(np.maximum(delta[unknown],0).sum())
        # Exhibit fixed truth assignments attaining each endpoint exactly.
        isfalse_low=np.zeros(len(a),bool);isfalse_high=np.zeros(len(a),bool)
        isfalse_low[list(false_ids)]=isfalse_high[list(false_ids)]=True
        isfalse_low[unknown & (delta<0)]=True
        isfalse_high[unknown & (delta>0)]=True
        assert float(delta[isfalse_low].sum())==low
        assert float(delta[isfalse_high].sum())==high
        return {'lower':low,'upper':high,'known_false_change':known,
                'unknown_negative_change_lower':low-known,'unknown_positive_change_upper':high-known,
                'certified_positive_drift_with_supported_gold':bool(gold_ok and low>0),
                'certified_no_positive_false_drift_with_supported_gold':bool(gold_ok and high<=0)}
    cases={
       'organization_occurrence_external_positives_only':bound(false_org,supported.get(sid,set())),
       'personal_action_occurrence_external_positives_only':bound(false_org|false_life,supported.get(sid,set())),
       'organization_occurrence_plus_benchmark_positive_soundness':bound(false_org,truths|supported.get(sid,set())),
       'personal_action_occurrence_plus_benchmark_positive_soundness':bound(false_org|false_life,truths|supported.get(sid,set()))}
    exits=np.flatnonzero(delta<0);cross=np.flatnonzero(delta>0)
    records.append({'sample_id':sid,'gold_fact':r['gold_fact'],'gold_independently_supported':gold_ok,
       'original_filtered_rank':r['original_gold_rank'],'final_filtered_rank':r['final_gold_rank'],
       'raw_rank_before':1+float(wb.sum()),'raw_rank_after':1+float(wa.sum()),
       'raw_positive_change_n':len(cross),'raw_negative_change_n':len(exits),
       'unchanged_strict_outrankers_cancel_n':int(((wa==1)&(wb==1)).sum()),
       'ties_before':int((wb==.5).sum()),'ties_after':int((wa==.5).sum()),
       'known_true_names':[names[k] for k in sorted(supported.get(sid,set()))],
       'organization_false_names':[names[k] for k in sorted(false_org)],
       'personal_action_additional_false_names':[names[k] for k in sorted(false_life)],
       'raw_exiting_candidate_names':[names[k] for k in exits],
       'bounds':cases})
keys=list(records[0]['bounds'])
summary={k:{'certified_positive_ids':[r['sample_id'] for r in records if r['bounds'][k]['certified_positive_drift_with_supported_gold']],
            'supported_gold_no_positive_drift_ids':[r['sample_id'] for r in records if r['bounds'][k]['certified_no_positive_false_drift_with_supported_gold']]} for k in keys}
out={'sample_n':40,'population_n':42851,'fixed_truth_across_checkpoints':True,
     'definition':'D_false=sum_e z_e*(w_after(e)-w_before(e)); z_e is fixed unknown falsity of the same dated query answer. Each weight is strict-above1/tie0.5/below0; absent before candidates contribute0.',
     'formula':'LB=sum_knownfalse(delta)+sum_unknown(min(delta,0)); UB=sum_knownfalse(delta)+sum_unknown(max(delta,0)); knowntrue contributes0.',
     'bound_scope':'Exact coordinate-wise logical bounds conditional on accepted labels, fixed dated truth, identity and candidate availability. Cross-query identity consistency can constrain attainable population combinations; no population point estimate.',
     'organization_scenario':'Prior dated named-organization/affiliation labels, at actual event occurrence date; excludes lifetime labels.',
     'personal_action_scenario':'Additionally requires dated new personal action by the deceased person; excludes retrospective reporting/posthumous publication as answers. This stricter interpretation is not established as ICEWS coder semantics.',
     'benchmark_soundness_scenario':'All existing exact-date benchmark positives also assumed worldtrue; independent external-only scenario does not require this.',
     'world_error_fraction':None,'world_reasonable_fraction':None,'summary':summary,'records':records,
     'training':False,'neural_rescoring':False,'seconds':time.perf_counter()-start}
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'seconds':out['seconds'],'summary':summary},ensure_ascii=False))
for r in records:
    if r['sample_id'] in ['S05','S08','S22','S29','S33','S38','S39']:
        print(json.dumps({k:v for k,v in r.items() if k!='bounds'},ensure_ascii=False))
        print(json.dumps(r['bounds'],ensure_ascii=False))
