"""Read-only, post-hoc transitions in already audited native rank exports.

No model scoring, training, candidate expansion or changes to old releases.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
assert not args.output.exists(), 'Never overwrite a saved analysis'
specs = []
for split in ['entity', 'relation', 'fact', 'hybrid']:
    jid = 'incde-' + split + '-s3407'
    specs.append((jid, 'IncDE', split.upper(), 'ranks', 'post_train',
                  root / 'results/KGE-LIT-001' / jid / 'audit.json', 'artifact_sha256'))
for method in ['fastkge', 'stark']:
    jid = method + '-fbckge-s3407'
    specs.append((jid, 'FastKGE' if method == 'fastkge' else 'STARK', 'FB_CKGE',
                  'native/seed3407/observations', 'after',
                  root / 'results/KGE-LIT-002' / jid / 'independent-audit.json', 'task_matrix'))
specs.append(('lkge-clean-fact-s55', 'LKGE_clean_validation', 'FACT', 'ranks',
              'post_train', root / 'results/KGE-LIT-006/independent-audit.json', 'artifact_sha256'))
records = []
bound = {}
for jid, method, split, subdir, phase, audit_path, map_key in specs:
    audit_digest = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    # 002 uses an explicit sampled-checkpoint replay status in its frozen report.
    assert audit['status'] in ['PASS', 'PASS_WITH_LIMITATIONS', 'PASS_SAMPLED_CHECKPOINT_REPLAY'], jid
    assert not audit.get('failures'), jid
    bound[str(audit_path.relative_to(root))] = audit_digest
    run = root / 'runs' / jid
    pairs = []
    for old in range(4):
        paths = [run / subdir / f'train{trained}_test{old}_{phase}.npz' for trained in [old, 4]]
        arrays = []
        for p in paths:
            digest = sha(p)
            rel = str(p.relative_to(run))
            expected = (audit[map_key][p.name]['sha256'] if map_key == 'task_matrix'
                        else audit[map_key][rel])
            assert digest == expected, rel
            bound[str(p.relative_to(root))] = digest
            with np.load(p, allow_pickle=False) as z:
                arrays.append({k: z[k].copy() for k in z.files})
        before, after = arrays
        for key in ['head', 'relation', 'tail']:
            assert np.array_equal(before[key], after[key]), (jid, old, key)
        candidate_key = 'candidate_count' if 'candidate_count' in before else 'candidates'
        assert np.array_equal(before[candidate_key], after[candidate_key]), (jid, old, 'candidates')
        if 'filtered_count' in before:
            assert np.array_equal(before['filtered_count'], after['filtered_count']), (jid, old, 'filter_count')
        if map_key == 'task_matrix':
            assert audit[map_key][paths[0].name]['filter_sha256'] == audit[map_key][paths[1].name]['filter_sha256']
        b, a = before['rank'], after['rank']
        assert b.shape == a.shape == before['head'].shape and b.ndim == 1
        assert np.issubdtype(b.dtype, np.integer) and np.issubdtype(a.dtype, np.integer)
        assert np.all(b >= 1) and np.all(a >= 1)
        counts = dict(queries=len(b), originally_top10=int(np.count_nonzero(b <= 10)),
            retained_top10=int(np.count_nonzero((b <= 10) & (a <= 10))),
            dropped_out_of_top10=int(np.count_nonzero((b <= 10) & (a > 10))),
            newly_entered_top10=int(np.count_nonzero((b > 10) & (a <= 10))),
            final_top10=int(np.count_nonzero(a <= 10)),
            exact_ranks_unchanged=int(np.count_nonzero(b == a)),
            ranks_worsened=int(np.count_nonzero(a > b)),
            ranks_improved=int(np.count_nonzero(a < b)))
        # Separate scalar arithmetic catches boundary/aggregation mistakes.
        scalar = dict.fromkeys(counts, 0)
        for old_rank, new_rank in zip(b.tolist(), a.tolist()):
            scalar['queries'] += 1
            scalar['originally_top10'] += int(old_rank <= 10)
            scalar['retained_top10'] += int(old_rank <= 10 and new_rank <= 10)
            scalar['dropped_out_of_top10'] += int(old_rank <= 10 and new_rank > 10)
            scalar['newly_entered_top10'] += int(old_rank > 10 and new_rank <= 10)
            scalar['final_top10'] += int(new_rank <= 10)
            scalar['exact_ranks_unchanged'] += int(old_rank == new_rank)
            scalar['ranks_worsened'] += int(new_rank > old_rank)
            scalar['ranks_improved'] += int(new_rank < old_rank)
        assert scalar == counts
        assert counts['originally_top10'] == counts['retained_top10'] + counts['dropped_out_of_top10']
        assert counts['final_top10'] == counts['retained_top10'] + counts['newly_entered_top10']
        assert counts['queries'] == counts['exact_ranks_unchanged'] + counts['ranks_worsened'] + counts['ranks_improved']
        pairs.append(dict(old_snapshot=old, final_training_snapshot=4,
            before_file=str(paths[0].relative_to(root)), after_file=str(paths[1].relative_to(root)),
            fixed_historical_candidates=True, **counts))
    totals = {k: sum(p[k] for p in pairs) for k in counts}
    totals['drop_fraction_of_original_top10'] = totals['dropped_out_of_top10'] / totals['originally_top10']
    records.append(dict(id=jid, method=method, split=split, cohorts=pairs, totals=totals))

for rel, digest in bound.items():
    assert sha(root / rel) == digest, 'Input changed during analysis: ' + rel
out = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='PASS_DERIVED_NATIVE_RANK_TRANSITIONS',
    design='Post-hoc descriptive analysis. Each old test cohort0..3 is paired from its first post-training rank export to final snapshot4, with identical exported query identities and historical candidate scopes.',
    threshold=10, runs=records, bound_input_sha256=bound,
    analyzer_sha256=sha(Path(__file__)), scalar_count_crosscheck=True,
    new_training=False, new_model_inference=False,
    limitations=[
        'Native filtered double-argsort rankings; not the unfiltered strict-competition single-edit Rank-Level Locality protocol.',
        'Historical candidate universes remain fixed; says nothing about protection against all new entities at the current snapshot.',
        'Query observations include head/tail directions and possible duplicates; not independent samples or unique facts.',
        'Single seed per run, old test/cohort exposure and upstream split limitations retained; no significance or global guarantee.',
        'SAGE omitted because original integrity FAIL is retained. KGEditor is a different editing task and is not in this analysis.',
        'Existing score/rank producer audit is consumed; this script independently checks derivations, not all model scores.'
    ])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps([dict(method=r['method'], split=r['split'], **r['totals']) for r in records], indent=2))
