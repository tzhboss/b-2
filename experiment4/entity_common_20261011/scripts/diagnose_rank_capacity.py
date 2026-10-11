"""Config-driven TRAIN-only capacity diagnosis; no model updates or utility reads."""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def run(root, cfg):
    from common_incremental import Data
    from common_training import sha
    from rank_capacity import saturated_mask, score_certificate, selfcheck
    from shared_rank_cache_v2 import fixed_scores
    started = time.monotonic(); torch.set_num_threads(cfg['cpu_threads'])
    task = json.loads((root/cfg['task_config']).read_text()); data = Data(root, task)
    old = data.triples(0, 'train').numpy(); new = data.triples(1, 'train').numpy()
    truth = defaultdict(set); new_facts = defaultdict(int)
    for h, r, t in old: truth[(int(h), int(r))].add(int(t))
    for h, r, t in new:
        if h < cfg['old_entities'] <= t: new_facts[(int(h), int(r))] += 1
    bound = {str(root/cfg['task_config']): sha(root/cfg['task_config']),
             str(Path(__file__)): sha(Path(__file__)),
             str(root/'src/rank_capacity.py'): sha(root/'src/rank_capacity.py')}
    rows = []; arrays = {}
    for seed in cfg['seeds']:
        scope = root/cfg['scope_directory']/f'shared-rank-scope-s{seed}'
        audit = json.loads((scope/'audit.json').read_text())
        receipt = json.loads((scope/'receipt.json').read_text())
        assert audit['status'].startswith('PASS') and not audit['failures']
        assert not receipt['test_read'] and not receipt['valid_read']
        assert audit['receipt_sha256'] == sha(scope/'receipt.json')
        for name, digest in receipt['input_sha256'].items(): assert sha(Path(name)) == digest, name
        assert sha(scope/'scope.npz') == receipt['output_sha256']['scope.npz']
        with np.load(scope/'scope.npz') as z:
            keys = z['source_keys'].copy(); members = z['pred_old_members'].copy()
            true = z['source_true'].copy(); tq = z['source_true_query_ids'].copy()
            tranks = z['source_true_initial_ranks'].copy()
        source = root/'artifacts/incremental'/cfg['source_template'].format(seed=seed)
        source_audit = json.loads((source/'audit.json').read_text())
        assert source_audit['status'].startswith('PASS') and not source_audit.get('failures')
        assert sha(source/'initial.pt') == receipt['input_sha256'][str(source/'initial.pt')]
        model = torch.load(source/'initial.pt', map_location='cpu', weights_only=True)
        E = model['E'][:cfg['old_entities']].double(); R = model['R'].double()
        assert E.shape[1] == cfg['dimension'] and members.shape[1] == cfg['k']
        assert torch.equal(E.float().double(), E) and torch.equal(R.float().double(), R)
        affected = np.asarray([tuple(q) in new_facts for q in keys])
        labels = np.asarray([sum(int(g) in truth.get(tuple(q), ()) for g in row)
                             for q, row in zip(keys, members)])
        saturated = saturated_mask(keys, members, truth) & affected
        chosen = np.flatnonzero(saturated); certified = []; records = []
        max_tree_error = 0.; min_gap = float('inf'); min_ratio = float('inf')
        for off in range(0, len(chosen), cfg['score_batch']):
            ids = chosen[off:off+cfg['score_batch']]; part = keys[ids]
            Q = E[part[:,0]] * R[part[:,1]]
            scores = fixed_scores(Q, E).numpy()
            sums = (Q.abs() @ E.abs().T).numpy()
            # Independently reduce each complete old candidate row in NumPy.
            for qi, saved, abs_sum in zip(ids, scores, sums):
                h, r = keys[qi]; direction = E[h].numpy() * R[r].numpy()
                padded = 1 << (cfg['dimension']-1).bit_length()
                tree = np.zeros((len(E), padded)); tree[:,:cfg['dimension']] = E.numpy() * direction
                while tree.shape[1] > 1: tree = tree[:,0::2] + tree[:,1::2]
                independent = tree[:,0]
                # The standard relative roundoff model assumes normal nonzero
                # products. Reject a certificate if any actual factor product
                # can fall in FP32 subnormal range (zeros remain exact).
                nonzero = np.abs(E.numpy() * direction)
                assert not np.any((nonzero > 0) & (nonzero < np.finfo(np.float32).tiny))
                assert np.array_equal(saved, independent), 'Independent full old-pool scores differ'
                max_tree_error = max(max_tree_error, float(np.max(np.abs(saved-independent))))
                actual = np.argsort(-independent, kind='stable')[:cfg['k']]
                assert np.array_equal(actual, members[qi]), 'Original scope member identity differs'
                assert all(int(g) in truth[tuple(keys[qi])] for g in actual)
                cert = score_certificate(independent, actual, abs_sum, cfg['dimension'])
                min_gap = min(min_gap, cert['strict_gap'])
                min_ratio = min(min_ratio, cert['strict_gap']/(2*cert['max_score_error_bound']))
                records.append([int(qi), cert['strict_gap'], cert['max_score_error_bound'], int(cert['robust_boundary'])])
                if cert['robust_boundary']: certified.append(int(qi))
            print('CAPACITY_DIAGNOSTIC', seed, off+len(ids), len(chosen), flush=True)
        certified = np.asarray(certified, dtype=np.int64)
        # Every known old gold outside a strict certified list has rank > k.
        outside_truth = 0
        for qi in certified:
            outside_truth += len(truth[tuple(keys[qi])] - set(map(int, members[qi])))
        selected_true = np.isin(tq, certified) & (tranks <= cfg['k'])
        row = dict(seed=seed, affected_source_queries=int(affected.sum()),
                   affected_old_top10_train_truth_histogram=np.bincount(labels[affected], minlength=cfg['k']+1).tolist(),
                   saturated_affected_queries=len(chosen), certified_saturated_queries=len(certified),
                   saturated_new_train_facts=sum(new_facts[tuple(keys[i])] for i in chosen),
                   certified_saturated_new_train_facts=sum(new_facts[tuple(keys[i])] for i in certified),
                   selected_initially_eligible_true_anchors_at_certified_queries=int(selected_true.sum()),
                   old_train_truths_outside_certified_top10=outside_truth,
                   old_candidate_scores_independently_replayed=len(chosen)*len(E),
                   max_numpy_torch_tree_error=max_tree_error,
                   minimum_top10_gap=min_gap if len(chosen) else None,
                   minimum_gap_to_error_ratio=min_ratio if len(chosen) else None)
        rows.append(row); arrays[f's{seed}_query_ids'] = certified
        arrays[f's{seed}_query_keys'] = keys[certified]
        arrays[f's{seed}_certificates'] = np.asarray(records)
        for p in [source/'initial.pt', source/'audit.json', scope/'scope.npz', scope/'receipt.json', scope/'audit.json']:
            bound[str(p)] = sha(p)
        bound.update(receipt['input_sha256'])
    bound.update(data.bound)
    for name, digest in bound.items(): assert sha(Path(name)) == digest, name
    out = root/cfg['output_directory']; out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'certificates.npz', **arrays)
    summary = dict(status='COMPLETED_EXPLORATORY_TRAIN_CAPACITY_DIAGNOSTIC',
                   test_read=False, valid_read=False, controls=selfcheck(), rows=rows,
                   elapsed_s=time.monotonic()-started, input_sha256=bound,
                   output_sha256=sha(out/'certificates.npz'),
                   implementation_sha256=sha(Path(__file__)), module_sha256=sha(root/'src/rank_capacity.py'),
                   limits=['Not a new method, utility gate or promoted scientific result.',
                           'TRAIN query selection uses only old/new TRAIN and original old candidate scores.',
                           'Strict boundary separates a full list of k known TRAIN answers from every other old candidate.',
                           'Disjoint unseen old gold cannot originally succeed at these keys under certified arithmetic; split disjointness remains an explicit assumption, not checked using TEST here.',
                           'Dropping constraints would change trained-truth preservation; no such intervention is executed.',
                           'Does not preserve old MRR outside Top10, certify all floating-point implementations, or prove new-answer utility.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--config', required=True)
    a = p.parse_args(); root = a.root.resolve(); sys.path.insert(0, str(root/'src'))
    run(root, json.loads((root/a.config).read_text()))
