"""Read cached Top1s only; dated-organization diagnostic is conditional evidence."""
import collections
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / 'temporal_drift76_20261014'
started = time.perf_counter()
mapping = json.loads((MODEL_ROOT / 'prepared/mapping.json').read_text())
with np.load(MODEL_ROOT / 'prepared/data.npz') as archive:
    test = archive['test']
evidence = json.loads((ROOT / 'representative-history-evidence.json').read_text())
sample = json.loads((ROOT / 'representative-history-sample.json').read_text())
sample_by_id = {r['test_directional_index']: r['sample_id'] for r in sample['records']}
supported_gold = {
    r['test_directional_index'] for r in sample['records']
    if evidence['judgments'][r['sample_id']]['gold_truth'] == 'supported_at_query_semantic_granularity'
}
R = len(mapping['relations'])
assert R == 251
# Cutoffs are conservative earliest possible years, not asserted exact founding days.
specs = [
    ('Aam Aadmi Party', 8641, '2012-01-01', ['AAP_FORMATION_OFFICIAL']),
    ('Free Egyptians Party', 8351, '2011-01-01', ['FREE_EGYPTIANS_LAUNCH']),
    ('March 23 Movement', 8457, '2012-01-01', ['M23_CANADA_HISTORY']),
    ('Terrorist (Jabhat al-Nusra)', 9647, '2011-01-01', ['NUSRA_USDOS_HISTORY', 'NUSRA_UN_HISTORY']),
    ('Combatant (Jabhat al-Nusra)', 8298, '2011-01-01', ['NUSRA_USDOS_HISTORY', 'NUSRA_UN_HISTORY']),
    ('Liberal National Party', 6404, '2007-01-01', ['LNP_FORMATION_OFFICIAL', 'LNP_AEC_REGISTRATION']),
    ('Jaysh al-Islam', 9321, '2011-01-01', ['JAYSH_PREDECESSOR_MMP']),
    ('Militant (Ahrar ash-Sham)', 9421, '2011-01-01', ['AHRAR_FORMATION_MMP']),
    ('Tasnim news agency', 9641, '2012-01-01', ['TASMIN_FORMATION_PRIMARY', 'TASMIN_ABOUT_PRIMARY']),
]
for name, idx, cutoff, source_ids in specs:
    assert mapping['entities'][idx] == name, (idx, mapping['entities'][idx], name)
    assert all(s in evidence['sources'] for s in source_ids)
spec_by_id = {s[1]: s for s in specs}
actual_target_relations = {
    'Consult', 'Discuss by telephone', 'Engage in negotiation',
    'Host a visit', 'Make a visit', "Meet at a 'third' location",
    'Sign formal agreement', 'Cooperate economically', 'Cooperate militarily',
    'Engage in diplomatic cooperation', 'Engage in judicial cooperation',
    'Engage in material cooperation', 'Share intelligence or information',
}
arms = {}
all_cases = []
for path in sorted((ROOT / 'icews').glob('*-history.npz')):
    arm = path.name.removesuffix('-history.npz')
    with np.load(path) as archive:
        a = {k: archive[k] for k in archive.files}
    ids = a['ids']
    rows = test[ids]
    dates = np.array([mapping['dates'][int(d)] for d in rows[:, 3]])
    role_actor = rows[:, 1] >= R
    relation_names = np.array([mapping['relations'][int(r) % R] for r in rows[:, 1]])
    actual_target = np.isin(relation_names, sorted(actual_target_relations))
    masks = {}
    for phase in ['before', 'after']:
        any_name_date = np.zeros(len(ids), dtype=bool)
        for name, idx, cutoff, source_ids in specs:
            any_name_date |= (a['prediction_' + phase] == idx) & (dates < cutoff)
        participant = any_name_date & (role_actor | actual_target)
        masks[phase] = {'name_date': any_name_date, 'participant': participant}
    worsened = a['final_rank'] > a['original_rank']
    # Unique final raw Top1 strictly above gold, with no possibly false candidates
    # above gold at the original checkpoint. Raw rule does not assume filter soundness.
    unique_wrong_top = masks['after']['participant'] & (a['after_top1_ties'] == 1) & (a['final_raw_rank'] > 1)
    raw_rank_witness = unique_wrong_top & (a['original_raw_rank'] == 1)
    filtered_rank_witness = unique_wrong_top & (a['original_rank'] == 1)
    arm_summary = {
        'historical_directional_n': int(len(ids)),
        'worsened_filtered_rank_n': int(worsened.sum()),
        'before_name_date_conflict_n': int(masks['before']['name_date'].sum()),
        'after_name_date_conflict_n': int(masks['after']['name_date'].sum()),
        'before_participant_conflict_n': int(masks['before']['participant'].sum()),
        'after_participant_conflict_n': int(masks['after']['participant'].sum()),
        'after_participant_actor_n': int((masks['after']['participant'] & role_actor).sum()),
        'after_participant_target_actual_interaction_n': int((masks['after']['participant'] & ~role_actor).sum()),
        'after_name_date_only_target_other_relation_n': int((masks['after']['name_date'] & ~masks['after']['participant']).sum()),
        'after_participant_and_worsened_n': int((masks['after']['participant'] & worsened).sum()),
        'new_participant_conflict_n': int((masks['after']['participant'] & ~masks['before']['participant']).sum()),
        'after_participant_recorded_positive_n': int((masks['after']['participant'] & a['after_top1_recorded_true']).sum()),
        'conditional_gold_valid_raw_whole_false_rank_drift_witness_n': int(raw_rank_witness.sum()),
        'conditional_gold_valid_and_filter_sound_whole_false_rank_drift_witness_n': int(filtered_rank_witness.sum()),
        'independently_supported_fixed_sample_gold_raw_witness_ids': [sample_by_id[int(ids[i])] for i in np.flatnonzero(raw_rank_witness) if int(ids[i]) in supported_gold],
        'by_entity_after_name_date': dict(collections.Counter(mapping['entities'][int(x)] for x in a['prediction_after'][masks['after']['name_date']])),
        'by_entity_after_participant': dict(collections.Counter(mapping['entities'][int(x)] for x in a['prediction_after'][masks['after']['participant']])),
    }
    if arm == 'TComplEx_finetune':
        assert len(ids) == 81772 and int(worsened.sum()) == 42851
    for i in np.flatnonzero(masks['after']['name_date']):
        h, r, t, d = map(int, rows[i])
        predicted = int(a['prediction_after'][i])
        if r < R:
            gold_fact = [mapping['entities'][h], mapping['relations'][r], mapping['entities'][t], dates[i]]
            predicted_fact = [gold_fact[0], gold_fact[1], mapping['entities'][predicted], dates[i]]
        else:
            gold_fact = [mapping['entities'][t], mapping['relations'][r - R], mapping['entities'][h], dates[i]]
            predicted_fact = [mapping['entities'][predicted], gold_fact[1], gold_fact[2], dates[i]]
        all_cases.append({
            'arm': arm, 'test_directional_index': int(ids[i]),
            'sample_id': sample_by_id.get(int(ids[i])),
            'gold_fact': gold_fact, 'final_top1_fact': predicted_fact,
            'predicted_role': 'actor' if role_actor[i] else 'target',
            'conservative_cutoff': spec_by_id[predicted][2],
            'source_ids': spec_by_id[predicted][3],
            'participant_conflict_under_identity_and_occurrence_semantics': bool(masks['after']['participant'][i]),
            'original_raw_rank': float(a['original_raw_rank'][i]), 'final_raw_rank': float(a['final_raw_rank'][i]),
            'original_filtered_rank': float(a['original_rank'][i]), 'final_filtered_rank': float(a['final_rank'][i]),
            'final_top1_ties': int(a['after_top1_ties'][i]),
            'final_top1_recorded_positive': bool(a['after_top1_recorded_true'][i]),
            'conditional_gold_valid_raw_false_rank_drift_witness': bool(raw_rank_witness[i]),
            'conditional_gold_valid_filter_sound_false_rank_drift_witness': bool(filtered_rank_witness[i]),
            'independent_gold_support_from_fixed40': int(ids[i]) in supported_gold,
        })
    arms[arm] = arm_summary

out = {
    'definition': 'Cached final raw Top1 named-organization predictions before conservative preexistence cutoff. Actor predictions and limited actual bilateral target interactions form conditional participant conflicts; other target relations remain weaker name-date flags.',
    'arms': arms,
    'organization_rules': [{'name': n, 'entity_id': i, 'cutoff_exclusive': c, 'source_ids': ss} for n, i, c, ss in specs],
    'actual_target_relation_allowlist': sorted(actual_target_relations),
    'conditions': [
        'Names denote independently dated organizations or affiliation at event date, not retrospective labels for predecessors or unrelated homonyms.',
        'Date denotes actual occurrence, not report date of a future or recalled event; relation denotes actual action at query semantic granularity.',
        'Raw-rank witness additionally assumes independently unverified benchmark gold valid; filtered witness also assumes recorded filtered positives valid.',
        'Only existing Top1 predictions examined; absence of flag is not evidence of temporal validity or reasonable ranking.',
        'Organization sources were targeted after exploratory inspection. Coverage counts are finite cached-dataset diagnostics, not representative true-error estimates.',
    ],
    'world_reasonable_fraction': None, 'world_error_fraction': None,
    'fixed40_replaced_or_enlarged': False,
    'training': False, 'neural_rescoring': False,
    'seconds': time.perf_counter() - started,
    'cost_scope': 'This cached-array read and classification only; excludes prior training/scoring, source research, coding, and mirroring.',
    'cases_artifact': 'cached-temporal-conflict-cases-round15.json',
}
(ROOT / 'cached-temporal-conflicts-round15.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
(ROOT / out['cases_artifact']).write_text(json.dumps(all_cases, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(out, ensure_ascii=False, indent=2))
