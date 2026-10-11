"""Synthetic shared-model derivative, safety, and redistribution controls."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np


def independent_ranks(E, R, guards):
    # Scalar loops do not reuse the producer query/scoring or receipt code.
    answer = []
    for g in guards:
        s = [sum(float(E[g.head, k]) * float(R[g.relation, k]) * float(E[c, k])
                 for k in range(E.shape[1])) for c in range(len(E))]
        answer.append(1 + sum(v > s[g.gold] for v in s))
    return answer


def run(root, cfg):
    sys.path.insert(0, str(root / 'src'))
    from shared_rank_redistribution import RankGuard, SharedGuardProblem, bounded_redistribution, rank_receipt
    rng = np.random.default_rng(cfg['seed']); cases = []; max_error = 0.
    # Head/gold/candidate aliasing and multiple simultaneously mutable rows.
    for case in range(cfg['jacobian_cases']):
        E = rng.normal(size=(13, 3)); R = rng.normal(size=(2, 3))
        ids = np.array([0, 1, 8, 9, 12]); witness = E.copy(); witness[ids] += rng.normal(size=(5, 3)) * .2
        guards = [RankGuard(0, 0, 9, 10), RankGuard(8, 1, 8, 10), RankGuard(9, 0, 0, 10)]
        p = SharedGuardProblem(E, R, ids, guards, witness)
        x = witness[ids].ravel(); actual = p.jacobian(x); numeric = np.zeros_like(actual)
        for k in range(len(x)):
            step = np.zeros_like(x); step[k] = cfg['difference_step']
            numeric[:, k] = (p.values(x + step) - p.values(x - step)) / (2 * cfg['difference_step'])
        error = float(np.max(np.abs(actual - numeric))); assert error < cfg['jacobian_tolerance'], (case, error)
        max_error = max(max_error, error)
        assert rank_receipt(witness, R, guards)['ranks'] == independent_ranks(witness, R, guards)
        cases.append(dict(case=case, max_abs_derivative_error=error))

    old = np.column_stack([np.arange(12, 0, -1), np.r_[np.arange(1, 10), 0, -1, -2]]).astype(float)
    E = np.vstack([old, [-1., -1.]]); R = np.array([[1/12, 0.], [0., 1.]])
    true = [RankGuard(0, 0, 9, 10), RankGuard(0, 1, 9, 10), RankGuard(8, 0, 9, 10), RankGuard(9, 0, 9, 10)]
    unaffected = [RankGuard(0, 1, i, 10, 'prediction') for i in range(10)]
    guards = true + unaffected
    initial_ranks = independent_ranks(E, R, guards); assert max(initial_ranks) <= 10
    source_hash = hashlib.sha256(E.tobytes() + R.tobytes()).hexdigest()
    # Fixed toy target; old representation penalty and old radius explicit.
    ids = [8, 12]; target = E[ids].copy(); target[1] = [5., -1.]
    radii = np.array([2., 7.]); weights = np.array([cfg['old_change_penalty'], 1.])
    actual, info = bounded_redistribution(E, R, ids, target, weights, radii, guards, cfg['solver'])
    assert info['accepted'], info
    assert np.array_equal(actual[np.setdiff1d(np.arange(len(E)), ids)], E[np.setdiff1d(np.arange(len(E)), ids)])
    assert hashlib.sha256(E.tobytes() + R.tobytes()).hexdigest() == source_hash
    ranks = independent_ranks(actual, R, guards); assert max(ranks) <= 10
    new_rank = independent_ranks(actual, R, [RankGuard(0, 0, 12, 10)])[0]; assert new_rank <= 10
    assert actual[8, 0] < E[8, 0] and actual[12, 0] > 3
    before_top = set(np.argsort(-((E[0] * R[1]) @ E.T), kind='stable')[:10])
    after_top = set(np.argsort(-((actual[0] * R[1]) @ actual.T), kind='stable')[:10])
    assert before_top == after_top == set(range(10))
    # This sufficient witness permits an explicitly disclosed affected-set swap.
    affected_before = set(np.argsort(-((E[0] * R[0]) @ E.T), kind='stable')[:10])
    affected_after = set(np.argsort(-((actual[0] * R[0]) @ actual.T), kind='stable')[:10])
    assert affected_before != affected_after and 12 in affected_after and 8 not in affected_after
    assert not np.array_equal((E[8] * R[0]) @ E.T, (actual[8] * R[0]) @ actual.T)
    assert not np.array_equal(E @ (E[0] * R[0]), actual @ (actual[0] * R[0]))

    # Same target with all old rows frozen: preserves old rank, cannot add new Top10.
    frozen, frozen_info = bounded_redistribution(E, R, [12], target[1:], [1.], [7.], guards, cfg['solver'])
    assert frozen_info['accepted'], frozen_info
    assert independent_ranks(frozen, R, [RankGuard(0, 0, 12, 10)])[0] > 10
    # Adding prediction protection to the edited query retains its old ten members;
    # solver cannot keep all of them and strict new success in a tie-free order.
    all_pred = guards + [RankGuard(0, 0, i, 10, 'prediction') for i in range(10)]
    all_model, all_info = bounded_redistribution(E, R, ids, target, weights, radii, all_pred, cfg['solver'])
    assert independent_ranks(all_model, R, [RankGuard(0, 0, 12, 10)])[0] > 10

    # Old frozen head/gold features can falsely approve changes to shared rows.
    bad_head = E.copy(); bad_head[0] = [0., 1.]
    stale_query = E[0] * R[0]; stale_gold = float(stale_query @ E[9])
    stale_new_safe = float(stale_query @ bad_head[12]) <= stale_gold
    assert stale_new_safe and independent_ranks(bad_head, R, [true[0]])[0] > 10
    bad_gold = E.copy(); bad_gold[9] = [-3., 0.]
    assert float(stale_query @ bad_gold[12]) <= stale_gold
    assert independent_ranks(bad_gold, R, [true[0]])[0] > 10

    fail_options = dict(cfg['solver'], maxiter=1)
    failed, fail_info = bounded_redistribution(E, R, ids, target, weights, radii, guards, fail_options)
    assert not fail_info['accepted'] and np.array_equal(failed, E)
    # A float-tolerance inequality is insufficient for an exact rank contract.
    ties = E.copy(); ties[12] = [3. + 1e-12, -1.]
    assert independent_ranks(ties, R, [true[0]])[0] == 11
    p = SharedGuardProblem(E, R, [12], [true[0]], E)
    assert np.min(p.values(ties[[12]].ravel())) > -cfg['solver']['primal_tolerance']
    assert not rank_receipt(ties, R, [true[0]])['passed']
    result = dict(passed=True, scope='Synthetic implementation and feasibility only; no graph utility or novelty claim',
                  test_read=False, graph_facts_read=False, jacobian_cases=len(cases), max_abs_derivative_error=max_error,
                  head_gold_candidate_aliasing_checked=True, scalar_full_rank_oracle=True,
                  shared_old_head_rescored=True, stale_head_false_certificate_rejected=True,
                  stale_gold_false_certificate_rejected=True, exact_tie_contract_checked=True,
                  unchanged_source_arrays=True, unaffected_prediction_set_preserved=True,
                  affected_prediction_set_changed=True, frozen_new_rank=independent_ranks(frozen, R, [RankGuard(0,0,12,10)])[0],
                  joint_new_rank=new_rank, old_guard_ranks=ranks, joint=info,
                  freeze_control=frozen_info, all_prediction_control=all_info,
                  forced_failure_returns_unchanged=True, cases=cases)
    out = root / cfg['output_directory']; out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'synthetic_models.npz', initial=E, redistributed=actual, frozen=frozen, relations=R)
    (out / 'receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--config', required=True)
    a = p.parse_args(); run(a.root.resolve(), json.loads((a.root / a.config).read_text()))
