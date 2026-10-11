"""Bounded same-score timing on audited TRAIN query keys; no graph updates."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, cfg, config_path):
    from adaptive_rank_scoring import adaptive_fixed_scores
    from shared_rank_cache_v2 import fixed_scores
    from shared_rank_graph import load_scope

    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    canonical = yaml.safe_load((root/cfg['source_canonical']).read_text())
    assert canonical['jobs'][cfg['source_job']]['status'] == 'completed_audited'
    source = root/'artifacts/incremental'/cfg['source_job']
    receipt = json.loads((source/'training-receipt.json').read_text())
    audit = json.loads((source/'audit.json').read_text())
    assert not receipt['test_read'] and not audit['failures']
    assert audit['status'] == 'PASS_COMMON_INFERENCE_ALL3753_PATCHES_AND_FULL_TERMINAL_SCOPE'
    assert digest(source/'initial.pt') == receipt['output_sha256']['initial.pt'] == cfg['initial_sha256']
    keys, refs, bound = load_scope(root, cfg, cfg['seed'], 'full')
    scope_path = root/cfg['scope_directory']/f"shared-rank-scope-s{cfg['seed']}"/'scope.npz'
    assert digest(scope_path) == cfg['scope_sha256']
    bound.update({str(source/name): digest(source/name) for name in ['initial.pt', 'training-receipt.json', 'audit.json']})
    bound[str(config_path)] = digest(config_path)
    model = torch.load(source/'initial.pt', map_location=device, weights_only=True)
    E, R = model['E'], model['R']
    assert E.dtype == R.dtype == torch.float64 and E.shape == (11258, 200)
    key_tensor = torch.as_tensor(keys, dtype=torch.long, device=device)
    Q = E[key_tensor[:, 0]] * R[key_tensor[:, 1]]
    assert len(Q) == cfg['expected_queries']
    measurements = []
    started = time.monotonic()

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    for ids in cfg['candidate_ids']:
        rows = E[ids]
        expected = fixed_scores(Q, rows)
        actual = adaptive_fixed_scores(Q, rows)
        assert torch.equal(expected.view(torch.int64), actual.view(torch.int64))
        # Every score is also checked against a CPU NumPy reduction tree.
        q, e = Q.cpu().numpy(), rows.cpu().numpy()
        scores = actual.cpu().numpy()
        padded = 1 << (E.shape[1]-1).bit_length()
        for offset in range(0, len(q), cfg['numpy_query_batch']):
            part = q[offset:offset+cfg['numpy_query_batch']]
            tree = np.zeros((len(part), len(e), padded), dtype=np.float64)
            tree[:, :, :E.shape[1]] = part[:, None, :] * e[None, :, :]
            while tree.shape[-1] > 1:
                tree = tree[..., 0::2] + tree[..., 1::2]
            assert np.array_equal(tree[..., 0].copy().view(np.uint64), scores[offset:offset+len(part)].copy().view(np.uint64))
        timings = {'original': [], 'adaptive': []}
        functions = {'original': fixed_scores, 'adaptive': adaptive_fixed_scores}
        for name, function in functions.items():
            function(Q, rows)
            synchronize()
        for repeat in range(cfg['timing_repeats']):
            order = ['original', 'adaptive'] if repeat % 2 == 0 else ['adaptive', 'original']
            for name in order:
                synchronize()
                start = time.perf_counter()
                result = functions[name](Q, rows)
                synchronize()
                timings[name].append(time.perf_counter()-start)
                assert torch.equal(result.view(torch.int64), expected.view(torch.int64))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        measurements.append(dict(candidate_ids=ids, queries=len(Q), score_cells=len(Q)*len(ids),
                                 bitwise_original_and_numpy=True, seconds=timings, median_seconds=medians,
                                 median_speed_ratio=medians['original']/medians['adaptive']))
        print('REAL_TRAIN_SCORING', device, len(Q), ids, medians, flush=True)
    for name, value in bound.items():
        assert digest(Path(name)) == value
    result = dict(status='PASS_REAL_TRAIN_FIXED_TREE_SCORING_BIT_PATTERNS', device=str(device),
                  torch_version=torch.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  query_groups=len(keys), reference_members=len(refs), measurements=measurements,
                  elapsed_s=time.monotonic()-started, input_sha256=bound,
                  code_sha256={name: digest(root/name) for name in ['src/adaptive_rank_scoring.py', 'src/shared_rank_cache_v2.py', 'scripts/check_real_rank_scoring.py']},
                  test_scored=False, valid_scored=False, model_updates=0,
                  limits=['TRAIN query-key score and narrow-call timing only; not complete trajectory equivalence.',
                          'One device/session and fixed initial rows; concurrency affects timings; no end-to-end speed claim.',
                          'No current registered source/config/scorer, scientific gate or model is modified.'])
    destination = root/cfg['output_directory']
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'summary.json').write_text(json.dumps(result, indent=2))
    print(result['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    sys.path.insert(0, str(root/'src'))
    config_path = root/arguments.config
    run(root, json.loads(config_path.read_text()), config_path)
