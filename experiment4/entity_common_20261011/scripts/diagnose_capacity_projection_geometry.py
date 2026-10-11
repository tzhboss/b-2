"""Exploratory parameter geometry on already audited seed17 endpoints only."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def main(root, output_name):
    torch.set_num_threads(2)
    table_path = root / 'tables/common-comparison-v4/table.json'
    table = json.loads(table_path.read_text())
    methods = {r['method']: r for r in table['rows'] if r['model'] == 'DistMult'}
    source_dir = root / 'artifacts/incremental/newentityonly-distmult-s17'
    source_receipt = json.loads((source_dir / 'training-receipt.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source_dir / 'best.pt') == source_receipt['checkpoint_sha256']
    target_model = torch.load(source_dir / 'best.pt', map_location='cpu', weights_only=True)
    initial_model = torch.load(source_dir / 'initial.pt', map_location='cpu', weights_only=True)
    target, initial = target_model['E'].double().numpy(), initial_model['E'].double().numpy()
    assert np.array_equal(target[:7505], initial[:7505])
    assert torch.equal(target_model['R'], initial_model['R'])
    bound = {str(p.relative_to(root)): digest(p) for p in
             (table_path, source_dir / 'best.pt', source_dir / 'initial.pt', source_dir / 'training-receipt.json')}
    models, rows = {}, []
    for name in ('SharedFrozenFull', 'CapacityFrozenFull', 'SharedFrozenScoped', 'CapacityFrozenScoped'):
        slot = next(s for s in methods[name]['per_seed_status'] if s['seed'] == 17)
        assert slot['status'] == 'completed_audited'
        folder = root / 'artifacts/incremental' / slot['id']
        tr = json.loads((folder / 'training-receipt.json').read_text())
        assert not tr['test_read'] and tr['processed_new_entities'] == 3753
        assert digest(folder / 'best.pt') == tr['checkpoint_sha256']
        assert tr['bound_input_sha256'][str(source_dir / 'best.pt')] == digest(source_dir / 'best.pt')
        model = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        E = model['E'].double().numpy()
        assert np.array_equal(E[:7505], initial[:7505])
        assert torch.equal(model['R'].double(), initial_model['R'].double())
        steps = json.loads((folder / 'steps.json').read_text())
        assert len(steps) == 3753
        assert sorted(s['entity'] for s in steps) == list(range(7505, 11258))
        assert all(s['mutable_ids'] == [s['entity']] for s in steps)
        assert all(s['receipt']['accepted'] for s in steps)
        with np.load(folder / 'shared-replay.npz') as replay:
            assert np.array_equal(replay['entity_order'], [s['entity'] for s in steps])
            assert np.array_equal(replay['after'][:, 0], E[replay['entity_order']])
        radii = np.asarray([s['radii'][0] for s in steps])
        order = np.asarray([s['entity'] for s in steps])
        movement = np.linalg.norm(E[order] - initial[order], axis=1)
        ratio = movement / radii
        assert (movement <= radii + 1e-8).all()
        target_ratio = np.linalg.norm(target[order] - initial[order], axis=1) / radii
        assert (target_ratio < 1).all()
        assert all(s['receipt']['history'][0]['bounds_ok'] for s in steps)
        projected = [s for s in steps if s['receipt']['status'] != 'direct_target']
        assert all(s['receipt']['history'][0]['violating_queries'] > 0 for s in projected)
        residual = np.linalg.norm(E[7505:] - target[7505:], axis=1)
        statuses = Counter(s['receipt']['status'] for s in steps)
        models[name] = E
        rows.append(dict(method=name, seed=17, processed_entities=3753,
                         accepted_updates=tr['accepted_updates'], zero_fallbacks=tr['zero_fallbacks'],
                         step_status_counts=dict(statuses),
                         source_squared_distance=float(np.square(residual).sum()),
                         exact_source_rows=int(np.all(E[7505:] == target[7505:], axis=1).sum()),
                         max_absolute_movement_to_radius_ratio=float(ratio.max()),
                         source_target_max_movement_to_radius_ratio=float(target_ratio.max()),
                         source_target_bounds_ok_all_steps=True,
                         projected_steps_initially_violate_rank_guards=len(projected),
                         rows_within_one_percent_of_radius=int((ratio >= 0.99).sum()),
                         old_parameters_exactly_unchanged=True))
        for filename in ('best.pt', 'training-receipt.json', 'steps.json', 'shared-replay.npz'):
            path = folder / filename
            bound[str(path.relative_to(root))] = digest(path)
    pairs = []
    for scope in ('Full', 'Scoped'):
        a, b = models['SharedFrozen' + scope][7505:], models['CapacityFrozen' + scope][7505:]
        old_distance = np.square(a - target[7505:]).sum(1)
        new_distance = np.square(b - target[7505:]).sum(1)
        changed = np.any(a != b, axis=1)
        pairs.append(dict(scope=scope, seed=17, changed_new_rows=int(changed.sum()),
                          source_distance_improved_rows=int((new_distance < old_distance).sum()),
                          source_distance_worsened_rows=int((new_distance > old_distance).sum()),
                          max_row_change_l2=float(np.linalg.norm(b-a, axis=1).max()),
                          total_source_squared_distance_delta=float((new_distance-old_distance).sum())))
    output = root / output_name
    assert not output.exists()
    result = dict(status='PASS_EXPLORATORY_AUDITED_SEED17_PARAMETER_GEOMETRY', rows=rows, pairs=pairs,
                  source_sha256=bound, optimization_or_training=False, new_valid_or_test_metrics_read=False,
                  registered_utility_decision=False,
                  limits=['Single-seed endpoint diagnostic; parameter distance is not retrieval utility.',
                          'Remaining rank constraints are not individually attributed causally here.',
                          'No active source, scope, update order, norm budget or gate changed.'])
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], rows=rows, pairs=pairs)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', default='results/capacity-partial-seed17-projection-geometry.json')
    args = parser.parse_args()
    main(args.root.resolve(), args.output)
