"""Independent saved-model scalar replay and analytic toy optimum audit."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, cfg):
    canonical_path = root / 'experiments' / cfg['id'] / 'experiment.yaml'
    canonical = yaml.safe_load(canonical_path.read_text())
    assert canonical['status'] == 'completed_pending_independent_check'
    for path, digest in canonical['frozen_input_sha256'].items():
        assert sha(root / path) == digest, path
    folder = root / cfg['output_directory']; receipt = json.loads((folder / 'receipt.json').read_text())
    with np.load(folder / 'synthetic_models.npz', allow_pickle=False) as arrays:
        E = arrays['initial']; changed = arrays['redistributed']; frozen = arrays['frozen']; R = arrays['relations']
    assert E.shape == changed.shape == frozen.shape == (13, 2) and R.shape == (2, 2)
    expected = np.array([[12-i, (i+1 if i < 9 else 9-i)] for i in range(12)] + [[-1, -1]], dtype=float)
    assert np.array_equal(E, expected) and np.array_equal(frozen, E)
    assert np.array_equal(R, [[1/12, 0], [0, 1]])
    scope = [8, 12]; unchanged = [i for i in range(13) if i not in scope]
    assert np.array_equal(changed[unchanged], E[unchanged])
    def row_scores(model, head, relation):
        return [sum(float(model[head, k]) * float(R[relation, k]) * float(model[c, k])
                    for k in range(2)) for c in range(13)]
    def rank(model, h, r, gold):
        s = row_scores(model, h, r)
        return 1 + sum(v > s[gold] for v in s)
    guards = [(0,0,9),(0,1,9),(8,0,9),(9,0,9)] + [(0,1,i) for i in range(10)]
    observed = [rank(changed, *g) for g in guards]
    assert observed == receipt['old_guard_ranks'] and max(observed) <= 10
    assert rank(E,0,0,12) == rank(frozen,0,0,12) == 13
    assert rank(changed,0,0,12) == receipt['joint_new_rank'] == 8
    def membership(model, h, r):
        s = row_scores(model,h,r)
        return set(sorted(range(13),key=lambda i:(-s[i],i))[:10])
    assert membership(E,0,1) == membership(changed,0,1) == set(range(10))
    assert membership(E,0,0) != membership(changed,0,0)
    assert 12 in membership(changed,0,0) and 8 not in membership(changed,0,0)
    assert row_scores(E,8,0) != row_scores(changed,8,0)
    movement = np.sqrt(np.sum((changed[scope]-E[scope])**2,axis=1))
    assert (movement <= np.array([2.,7.])+cfg['solver']['movement_tolerance']).all()
    # In this positive 2D toy, strongest R0 constraint is the smaller of
    # head9=3 and changed head8=x: x(3-x)/12 >= margin. This gives the upper
    # feasible root below3. Minimizing distance to old x=4 saturates that root;
    # target new x=5 and all second coordinates need no change.
    margin = cfg['solver']['guard_margin']
    root_x = (3. + np.sqrt(9.-48.*margin))/2.
    expected_rows = np.array([[root_x,9.],[5.,-1.]])
    analytic_error = float(np.max(np.abs(changed[scope]-expected_rows)))
    assert analytic_error < 1e-8, analytic_error
    target = np.array([[4.,9.],[5.,-1.]])
    objective = .5*sum(w*sum(float(a-b)**2 for a,b in zip(row,goal))
                       for w,row,goal in zip([cfg['old_change_penalty'],1.],changed[scope],target))
    assert abs(objective-receipt['joint']['objective']) < 1e-10
    assert receipt['jacobian_cases'] == cfg['jacobian_cases'] == len(receipt['cases'])
    assert receipt['max_abs_derivative_error'] == max(c['max_abs_derivative_error'] for c in receipt['cases'])
    assert receipt['max_abs_derivative_error'] < cfg['jacobian_tolerance']
    assert receipt['forced_failure_returns_unchanged'] and not receipt['freeze_control']['accepted']
    assert not receipt['test_read'] and not receipt['graph_facts_read']
    failed_path = root/'experiments'/'KGE-ENTITY-COMMON-SHARED-RANK-CONTROLS-022'/'experiment.yaml'
    failed = yaml.safe_load(failed_path.read_text())
    assert failed['status'] == 'failed_synthetic_controls_retained' and failed['exit_code'] == 1
    for path,digest in failed['frozen_input_sha256'].items(): assert sha(root/path) == digest
    audit = dict(status='PASS_SAVED_SHARED_MODEL_SCALAR_REPLAY_AND_ANALYTIC_TOY', failures=[],
                 old_guards_replayed=len(guards), new_gold_rank=rank(changed,0,0,12),
                 analytic_toy_max_vector_error=analytic_error, objective=objective,
                 unaffected_full_top10_membership=True, changed_head_rescored=True,
                 movement_bounds_checked=True, original_failed_control_preserved=True,
                 frozen_inputs_sha256_checked=True,
                 bound_artifact_sha256={name:sha(folder/name) for name in ['receipt.json','synthetic_models.npz']},
                 implementation_sha256=sha(Path(__file__)),
                 limits=['Synthetic only, not graph utility or novelty.',
                         'Analytic optimum check covers this constructed toy only.',
                         'Derivative controls executed by producer, saved-model replay independently scored.'])
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))
    formal=root/'results'/cfg['id'];formal.mkdir(parents=True,exist_ok=False)
    (formal/'summary.json').write_text(json.dumps(dict(result=receipt,audit=audit),indent=2))
    canonical.update(status='completed_audited_synthetic_controls_only',result_directory=str(formal.relative_to(root)))
    canonical_path.write_text(yaml.safe_dump(canonical,sort_keys=False))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
