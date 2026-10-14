"""Persist lifetime semantics and a conservative fixed-sample population bound."""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
started = time.perf_counter()
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

sample = read('representative-history-sample.json')
e = read('representative-history-evidence.json')
bounds = read('false-rank-drift-bounds-round10.json')
crossings = {r['sample_id']: r for r in read('fixed40-new-outrankers.json')['records']}
identity = read('lifetime-identity-context-round12.json')
assert sample['sample_n'] == 40 and sample['population_n'] == 42851
assert bounds['raw_confirmed_query_ids'] == ['S38']
e['sources'].update({
    'R12_CHIANG_OFFICIAL_LIFETIME': {
        'url': 'https://english.president.gov.tw/Page/83',
        'read_status': 'official_biography_body_read',
        'death_date': '1975-04-05',
        'evidence_summary': 'Official timeline explicitly dates death. Rules out new personal utterance in 2014, conditional on physical occurrence semantics; does not rule out contemporary reporting of historical words.'
    },
    'R12_RABIN_IDF_LIFETIME': {
        'url': 'https://www.idf.il/en/mini-sites/past-chiefs-of-staff/lt-gen-yitzhak-rabin-1964-1968/',
        'read_status': 'official_IDF_biography_body_read',
        'death_date': '1995-11-04',
        'evidence_summary': 'Official body explicitly dates killing November4,1995. Conditional on actual personal event occurrence, cannot personally consult or issue new praise in 2005. Reporting/quotation semantics unresolved.'
    },
    'R12_RAW_POSTHUMOUS_CONTEXT': {
        'artifact': 'lifetime-identity-context-round12.json',
        'read_status': 'original_coded_metadata_read_not_article_bodies',
        'evidence_summary': 'First5 available 2005 rows per actor include Chiang/Mao Consult April27 and Rabin/Arafat Signformalagreement February2. Establishes posthumous event coding exists, not which original sentence or failure produced it. No article-level identity proof.'
    }
})
lifetime_records = []
for sid, name, source in [
    ('S33', 'Chiang Kai-shek', 'R12_CHIANG_OFFICIAL_LIFETIME'),
    ('S22', 'Yitzhak Rabin', 'R12_RABIN_IDF_LIFETIME'),
    ('S29', 'Yitzhak Rabin', 'R12_RABIN_IDF_LIFETIME')
]:
    row = crossings[sid]
    candidate = next(c for c in row['new_outrankers'] if name in c['fact'])
    assert candidate['after_gap'] > 0 and candidate['before_gap'] < 0
    j = e['judgments'][sid]
    j['evidence_ids'] = list(dict.fromkeys(j['evidence_ids'] + [source, 'R12_RAW_POSTHUMOUS_CONTEXT']))
    j['lifetime_candidate_note'] = (
        'Cached pairwise crossing by a deceased actor. Negative only under new personal action at the query date; '
        'reported historical-content semantics unresolved. Does not change finalTop1 truth or prove positive total false-rank drift.'
    )
    lifetime_records.append({
        'sample_id': sid, **candidate, 'source_id': source,
        'conditional_personal_occurrence_truth': 'contradicted_after_death',
        'reported_historical_content_truth': 'unknown',
        'unconditional_temporal_negative': False,
        'gold_supported': j['gold_truth'] == 'supported_at_query_semantic_granularity',
        'whole_error_rank_drift_newly_proven': False
    })
write('lifetime-crossing-interpretation-round12.json', {
    'records': lifetime_records,
    'raw_metadata_scan_seconds': identity['seconds'],
    'unconditional_new_negative_n': 0,
    'new_whole_error_rank_drift_n': 0,
    'note': 'S33 before raw outranker upper is2: one conditional negative cannot force a positive total false-rank drift. S22/S29 golds remain unknown.',
    'training': False, 'neural_rescoring': False
})

# Fixed-size simple random sample; unknown cases never count as negative.
# If X_true >= X_certified, inversion at X_certified is conservative.
N, n, certified, alpha = sample['population_n'], sample['sample_n'], 1, 0.05
def at_least_one(K):
    if K > N - n:
        return 1.0
    return -math.expm1(sum(math.log1p(-K / (N - i)) for i in range(n)))
lo, hi = 0, N
while lo < hi:
    mid = (lo + hi) // 2
    if at_least_one(mid) >= alpha:
        hi = mid
    else:
        lo = mid + 1
K_lower = lo
assert at_least_one(K_lower - 1) < alpha <= at_least_one(K_lower)
population_bound = {
    'sampling_population': '42851 historical directional queries with worsened benchmark-filtered gold rank; TComplEx fine-tune only',
    'population_n': N, 'sample_n': n, 'certified_whole_false_rank_drift_n': certified,
    'certified_query_ids': ['S38'], 'confidence_level': 1-alpha,
    'method': 'One-sided finite-population hypergeometric tail inversion at at least1 true case; fixed random40 sample. Uses no independent-query/binomial assumption.',
    'formula': 'L=min{K:1-C(N-K,n)/C(N,n)>=0.05}; incomplete sound positive labels satisfy X_certified<=X_true.',
    'error_query_count_lower': K_lower,
    'error_fraction_among_worsened_query_lower': K_lower/N,
    'error_fraction_among_worsened_query_upper': 1.0,
    'population_point_error_fraction': None,
    'population_point_reasonable_fraction': None,
    'reasonable_fraction_upper_if_disjoint_from_positive_false_rank_drift': 1-K_lower/N,
    'boundary_tail_previous': at_least_one(K_lower-1),
    'boundary_tail_selected': at_least_one(K_lower),
    'all_historical_directional_queries_n': 81772,
    'error_fraction_among_all_historical_queries_lower': K_lower/81772,
    'conditions': bounds['conditions'] + [
        'S38 false-rank labels are sound at actual occurrence date and named organization semantics; no allowance for later reporting of future actors as earlier events',
        'Original seeded sample is retained without source-based replacement or enlargement',
        'Posthoc exploratory diagnostic of this fixed dataset/model; no cross-model/dataset generalization or correction of uncertain identity/source labels by sampling statistics'
    ],
    'unknown_treatment': '39 unresolved whole-rank cases may all be true errors or may all be nonerrors; never coded as reasonable.',
    'distinguish': 'Finite40 logical lower2.5% and population95% confidence lower have different meanings. Confidence lower is not a point estimate.',
    'training': False, 'neural_rescoring': False,
    'computation_seconds': time.perf_counter()-started
}
write('population-error-lower-bound-round12.json', population_bound)
e['round12_population_bound_artifact'] = 'population-error-lower-bound-round12.json'
write('representative-history-evidence.json', e)
assert sum(j['gold_truth'] == 'supported_at_query_semantic_granularity' for j in e['judgments'].values()) == 9
assert sum(j['final_event_truth'] == 'supported_at_query_semantic_granularity' for j in e['judgments'].values()) == 2
print(json.dumps(population_bound, ensure_ascii=False))
