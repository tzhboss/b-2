"""Exploratory paired TEST rank disclosure for already audited seed17 only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main(root):
    table = json.loads((root / 'tables/common-comparison-v4/table.json').read_text())
    methods = {r['method']: r for r in table['rows'] if r['model'] == 'DistMult'}
    names = ['NewEntityOnly', 'SharedFrozenFull', 'SharedFrozenScoped',
             'CapacityFrozenFull', 'CapacityFrozenScoped']
    bound = {}
    ranks = {}
    metrics = {}
    before_arithmetic = {}
    for name in names:
        row = methods[name]
        item = next(r for r in row['per_seed_status'] if r['seed'] == 17)
        assert item['status'] == 'completed_audited'
        path = root / 'artifacts/incremental' / item['id'] / 'ranks.npz'
        bound[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as raw:
            ranks[name] = {k: raw[k].copy() for k in ('old_before', 'old_after', 'new_after', 'old_queries', 'new_queries')}
        data = ranks[name]
        reference = ranks['NewEntityOnly']
        for key in ('old_queries', 'new_queries'):
            assert np.array_equal(data[key], reference[key]), (name, key)
        assert np.array_equal(data['old_before'] <= 10, reference['old_before'] <= 10)
        before_arithmetic[name] = dict(
            rank_differences_from_source=int((data['old_before'] != reference['old_before']).sum()),
            max_absolute_rank_difference=int(np.abs(data['old_before'] - reference['old_before']).max()),
            original_top10_eligibility_identical=True)
        assert len(data['old_after']) == 74428 and len(data['new_after']) == 12406
        query = data['new_queries']
        mask = (query[:, 0] < 7505) & (query[:, 2] >= 7505)
        assert int(mask.sum()) == 5654
        successful = data['old_before'] <= 10
        i = row['audited_seeds'].index(17)
        mrr = float((1 / data['new_after']).mean())
        loss = float((data['old_after'][successful] > 10).mean())
        assert abs(mrr - row['metrics']['new_mrr']['per_seed'][i]) < 1e-14
        assert loss == row['metrics']['old_top10_loss']['per_seed'][i]
        metrics[name] = dict(new_mrr=mrr, old_top10_loss=loss,
                             directional_mrr=float((1 / data['new_after'][mask]).mean()),
                             directional_hits10_count=int((data['new_after'][mask] <= 10).sum()),
                             directional_queries=int(mask.sum()))
    pairs = []
    for scope in ('Full', 'Scoped'):
        before = 'SharedFrozen' + scope
        after = 'CapacityFrozen' + scope
        a, b = ranks[before], ranks[after]
        assert np.array_equal(a['old_before'], b['old_before'])
        pairs.append(dict(original=before, candidate=after, seed=17,
                          new_mrr_delta=metrics[after]['new_mrr'] - metrics[before]['new_mrr'],
                          old_top10_loss_delta=metrics[after]['old_top10_loss'] - metrics[before]['old_top10_loss'],
                          directional_mrr_delta=metrics[after]['directional_mrr'] - metrics[before]['directional_mrr'],
                          directional_hits10_count_delta=metrics[after]['directional_hits10_count'] - metrics[before]['directional_hits10_count'],
                          new_ranks_improved=int((b['new_after'] < a['new_after']).sum()),
                          new_ranks_worsened=int((b['new_after'] > a['new_after']).sum()),
                          old_ranks_improved=int((b['old_after'] < a['old_after']).sum()),
                          old_ranks_worsened=int((b['old_after'] > a['old_after']).sum())))
    result = dict(status='PASS_PAIRED_AUDITED_SEED17_TEST_RANK_DISCLOSURE_ONLY',
                  seed=17, metrics=metrics, pairs=pairs, source_sha256=bound,
                  source_before_rank_arithmetic=before_arithmetic,
                  exploratory=True, registered_gate_decision=False,
                  limits=['Single audited seed; no VALID or TRAIN cost computation.',
                          'No change to active jobs, selection, budget, gates or update order.',
                          'Three-seed utility still requires completed038.'])
    path = root / 'results/capacity-partial-seed17-paired-ranks.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], pairs=pairs)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve())
