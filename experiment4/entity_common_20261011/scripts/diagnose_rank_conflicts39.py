"""TRAIN-only complete endpoint slot diagnosis and separate CPU fixtures."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common_incremental import Data
from shared_rank_graph import load_scope
from shared_rank_scope import source_initial
from shared_rank_cache_v2 import fixed_scores


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controls():
    # One target adds a competitor; a current competitor must be removed first.
    count = lambda o, n, a, b: o + n - int(a) + int(b)
    assert count(9, 0, False, True) == 10
    assert count(8, 1, False, True) == 10
    assert count(8, 1, True, True) == 9
    assert count(9, 0, False, False) == 9
    assert not (1.0 > 1.0)
    rng = np.random.default_rng(319)
    for case in range(128):
        old = rng.integers(-5, 6, 8).astype(float)
        members = np.argsort(-old, kind='stable')[:1+case%3]
        cut = old[members].min()
        actual = rng.integers(-5, 6, 4).astype(float)
        if (np.r_[old, actual] > cut).sum() >= 3:
            actual[:] = cut - 1
        desired = rng.integers(-5, 6, 4).astype(float)
        total = int((np.r_[old, actual] > cut).sum())
        for e in range(4):
            candidates = np.r_[old, actual].copy()
            candidates[8+e] = desired[e]
            ranks = [1+int((candidates > candidates[g]).sum()) for g in members]
            predicted = total-int(actual[e] > cut)+int(desired[e] > cut)
            assert (max(ranks) > 3) == (predicted >= 3)
    return dict(status='PASS_IMMUTABLE_AND_NEW_SLOT_SELF_REPLACEMENT_AND_TIE_CONTROLS',
                hand_controls=5, independently_enumerated_candidate_replacements=512)


def worker(root, cfg, seed):
    started = time.monotonic()
    torch.set_num_threads(2)
    source_cfg = json.loads((root / cfg['source_config']).read_text())
    device = torch.device('cuda:0')
    initial, R, bound = source_initial(root, source_cfg, seed, device)
    folder = root / 'artifacts/incremental' / source_cfg['source_template'].format(seed=seed)
    tr = json.loads((folder / 'training-receipt.json').read_text())
    assert not tr['test_read'] and sha(folder / 'best.pt') == tr['checkpoint_sha256']
    source = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
    target = source['E'].double()
    assert torch.equal(target[:7505], initial[:7505]) and torch.equal(source['R'].double(), R)
    bound.update({str(p): sha(p) for p in (folder / 'best.pt', folder / 'training-receipt.json')})
    data = Data(root, source_cfg)
    old = data.triples(0, 'train').numpy()
    new = data.triples(1, 'train').numpy()
    truth = set(map(tuple, old))
    affected = set(map(tuple, new[:, :2]))
    out = root / cfg['output_directory'] / f's{seed}'
    out.mkdir(parents=True, exist_ok=False)
    records = []
    fixture_data = {}
    for policy in cfg['policies']:
        endpoint = root / 'artifacts/incremental' / policy['endpoint'].format(seed=seed)
        audit = json.loads((endpoint / 'audit.json').read_text())
        solver = audit['solver_audit']
        assert not audit['failures'] and not solver['failures']
        assert solver['patches_replayed'] == 3753 and solver['terminal_cpu_counts_exact']
        receipt = json.loads((endpoint / 'training-receipt.json').read_text())
        assert sha(endpoint / 'best.pt') == receipt['checkpoint_sha256']
        current_model = torch.load(endpoint / 'best.pt', map_location=device, weights_only=True)
        current = current_model['E'].double()
        assert torch.equal(current[:7505], initial[:7505]) and torch.equal(current_model['R'].double(), R)
        scoped = dict(source_cfg, scope_directory=policy['scope_directory'])
        keys, refs, scope_bound = load_scope(root, scoped, seed, policy['mode'])
        bound.update(scope_bound)
        assert (keys[:, 0] < 7505).all() and (refs[:, 1] < 7505).all()
        scope_dir = root / policy['scope_directory'] / f'shared-rank-scope-s{seed}'
        with np.load(scope_dir / 'scope.npz') as raw:
            source_true = raw['source_true'].copy()
            eligible = raw['source_true_initial_ranks'] <= 10
        true_members = set(map(tuple, source_true[eligible]))
        per_query = [[] for _ in keys]
        for q, g in refs:
            per_query[int(q)].append(int(g))
        Q = initial[torch.as_tensor(keys[:, 0], device=device)] * R[torch.as_tensor(keys[:, 1], device=device)]
        cut = torch.full((len(keys),), float('inf'), dtype=torch.float64, device=device)
        for off in range(0, len(refs), 4096):
            part = refs[off:off+4096]
            values = (Q[torch.as_tensor(part[:, 0], device=device)] * initial[torch.as_tensor(part[:, 1], device=device)])
            tree = torch.zeros((len(values), 256), dtype=torch.float64, device=device)
            tree[:, :200] = values
            while tree.shape[1] > 1:
                tree = tree[:, 0::2] + tree[:, 1::2]
            cut.scatter_reduce_(0, torch.as_tensor(part[:, 0], device=device), tree[:, 0], reduce='amin', include_self=True)
        fixture_ids = np.sort(np.random.default_rng(cfg['fixture_seed']+seed).choice(len(keys), cfg['fixture_queries_per_policy'], replace=False))
        selected = set(map(int, fixture_ids))
        fixtures = []
        blocked = np.zeros(3753, dtype=bool)
        immutable = blocked.copy()
        competition = blocked.copy()
        histogram = np.zeros(10, dtype=np.int64)
        pair_counts = dict(immutable_old_capacity=0, new_candidate_slot_competition=0,
                           affected=0, unaffected=0, boundary_known_train=0, boundary_unlabelled=0)
        true_q = 0
        for qi, (h, r) in enumerate(keys):
            true_q += any((int(h), int(r), g) in true_members for g in per_query[qi])
        for off in range(0, len(keys), cfg['batch_queries']):
            assert time.monotonic()-started < cfg['budget_seconds_per_seed'], 'Registered diagnostic budget exceeded'
            q = Q[off:off+cfg['batch_queries']]
            old_scores = fixed_scores(q, initial[:7505])
            target_scores = fixed_scores(q, target[7505:])
            actual_scores = fixed_scores(q, current[7505:])
            threshold = cut[off:off+len(q)]
            old_count = (old_scores > threshold[:, None]).sum(1)
            actual_flag = actual_scores > threshold[:, None]
            desired_flag = target_scores > threshold[:, None]
            counts = old_count + actual_flag.sum(1)
            assert (counts < 10).all(), policy['method']
            violations = counts[:, None] - actual_flag.long() + desired_flag.long() >= 10
            hard = violations & (old_count[:, None] == 9)
            soft = violations & (old_count[:, None] < 9)
            bad = violations.cpu().numpy()
            blocked |= bad.any(0)
            immutable |= hard.any(0).cpu().numpy()
            competition |= soft.any(0).cpu().numpy()
            histogram += np.bincount(old_count.cpu().numpy(), minlength=10)
            pair_counts['immutable_old_capacity'] += int(hard.sum())
            pair_counts['new_candidate_slot_competition'] += int(soft.sum())
            old_np = old_scores.cpu().numpy()
            cut_np = threshold.cpu().numpy()
            for local, query in enumerate(keys[off:off+len(q)]):
                qi = off + local
                h, r = map(int, query)
                n = int(bad[local].sum())
                pair_counts['affected' if (h, r) in affected else 'unaffected'] += n
                lowest = [g for g in per_query[qi] if old_np[local, g] == cut_np[local]]
                assert lowest
                known = any((h, r, g) in truth for g in lowest)
                pair_counts['boundary_known_train' if known else 'boundary_unlabelled'] += n
                if qi in selected:
                    fixtures.append(dict(qi=qi, refs=per_query[qi], old=old_np[local],
                                         target=target_scores[local].cpu().numpy(), actual=actual_scores[local].cpu().numpy(),
                                         cut=cut_np[local], old_count=int(old_count[local]), count=int(counts[local]),
                                         blocked=bad[local]))
            if off % 8192 == 0:
                print('RANK_CONFLICT_PROGRESS', seed, policy['method'], off, len(keys), round(time.monotonic()-started, 2), flush=True)
        changed = (current[7505:] != target[7505:]).any(1).cpu().numpy()
        prefix = policy['method']
        for name in ('old', 'target', 'actual', 'blocked'):
            fixture_data[prefix+'_'+name] = np.asarray([f[name] for f in fixtures])
        for name in ('qi', 'cut', 'old_count', 'count'):
            fixture_data[prefix+'_'+name] = np.asarray([f[name] for f in fixtures])
        rows = dict(method=prefix, seed=seed, query_groups=len(keys), references=len(refs),
                    query_groups_with_selected_true_members=true_q,
                    query_groups_without_selected_true_members=len(keys)-true_q,
                    immutable_old_count_histogram=histogram.tolist(),
                    blocked_target_entities=int(blocked.sum()), immutable_old_blocked_entities=int(immutable.sum()),
                    new_slot_competition_blocked_entities=int(competition.sum()),
                    blocked_only_by_new_slot_entities=int((competition & ~immutable).sum()),
                    endpoint_rows_different_from_source=int(changed.sum()),
                    different_but_endpoint_source_replacement_feasible=int((changed & ~blocked).sum()),
                    blocking_pairs=pair_counts, fixture_query_ids=[f['qi'] for f in fixtures],
                    fixture_reference_members=[f['refs'] for f in fixtures])
        assert len(fixtures) == cfg['fixture_queries_per_policy']
        assert sum(histogram) == len(keys)
        total_pairs = pair_counts['immutable_old_capacity']+pair_counts['new_candidate_slot_competition']
        assert total_pairs == pair_counts['affected']+pair_counts['unaffected']
        assert total_pairs == pair_counts['boundary_known_train']+pair_counts['boundary_unlabelled']
        records.append(rows)
        for filename in ('best.pt', 'training-receipt.json', 'audit.json'):
            path = endpoint / filename
            bound[str(path)] = sha(path)
    bound.update(data.bound)
    for name, digest in bound.items():
        assert sha(Path(name)) == digest, name
    np.savez_compressed(out / 'fixtures.npz', **fixture_data)
    result = dict(status='COMPLETE_GPU_TRAIN_ONLY_ENDPOINT_SLOT_DIAGNOSIS', seed=seed, rows=records,
                  controls=controls(), input_sha256=bound, fixture_sha256=sha(out/'fixtures.npz'),
                  test_read=False, valid_read=False, weights_updated=False,
                  elapsed_s=time.monotonic()-started, gpu_name=torch.cuda.get_device_name(),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  limits=['Endpoint single-row feasibility does not prove order or global utility.',
                          'Unlabelled boundary member is not a proven false fact.',
                          'Independent CPU scores cover fixed32 queries per policy.'])
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('RANK_CONFLICT_COMPLETE', seed, round(result['elapsed_s'], 2), flush=True)


def tree_scores(direction, entities):
    tree = np.zeros((len(entities), 256))
    tree[:, :200] = entities * direction
    while tree.shape[1] > 1:
        tree = tree[:, 0::2] + tree[:, 1::2]
    return tree[:, 0]


def audit(root, cfg, seed):
    torch.set_num_threads(2)
    out = root / cfg['output_directory'] / f's{seed}'
    result = json.loads((out/'summary.json').read_text())
    assert not result['test_read'] and not result['valid_read'] and not result['weights_updated']
    for name, digest in result['input_sha256'].items():
        assert sha(Path(name)) == digest, name
    assert sha(out/'fixtures.npz') == result['fixture_sha256']
    source_cfg = json.loads((root/cfg['source_config']).read_text())
    initial, R, _ = source_initial(root, source_cfg, seed, 'cpu')
    initial, R = initial.numpy(), R.numpy()
    target = torch.load(root/'artifacts/incremental'/source_cfg['source_template'].format(seed=seed)/'best.pt', map_location='cpu', weights_only=True)['E'].double().numpy()
    cells = 0
    with np.load(out/'fixtures.npz') as fixtures:
        for policy, record in zip(cfg['policies'], result['rows']):
            assert record['method'] == policy['method']
            prefix = policy['method']
            current = torch.load(root/'artifacts/incremental'/policy['endpoint'].format(seed=seed)/'best.pt', map_location='cpu', weights_only=True)['E'].double().numpy()
            keys, refs, _ = load_scope(root, dict(source_cfg, scope_directory=policy['scope_directory']), seed, policy['mode'])
            assert np.array_equal(fixtures[prefix+'_qi'], record['fixture_query_ids'])
            for i, qi in enumerate(record['fixture_query_ids']):
                h, r = keys[qi]
                q = initial[h] * R[r]
                old = tree_scores(q, initial[:7505])
                desired = tree_scores(q, target[7505:])
                actual = tree_scores(q, current[7505:])
                for name, value in (('old', old), ('target', desired), ('actual', actual)):
                    assert np.array_equal(value.view(np.uint64), fixtures[prefix+'_'+name][i].view(np.uint64))
                    cells += len(value)
                members = refs[refs[:, 0] == qi, 1]
                assert np.array_equal(members, record['fixture_reference_members'][i])
                cut = min(old[members])
                old_count = int((old > cut).sum())
                count = old_count + int((actual > cut).sum())
                bad = count - (actual > cut).astype(np.int64) + (desired > cut).astype(np.int64) >= 10
                assert cut == fixtures[prefix+'_cut'][i]
                assert old_count == fixtures[prefix+'_old_count'][i] and count == fixtures[prefix+'_count'][i]
                assert count < 10 and np.array_equal(bad, fixtures[prefix+'_blocked'][i])
    audit_result = dict(status='PASS_SOURCE_IDENTITIES_AND_FIXED_FULL_QUERY_CPU_SLOT_FIXTURES', failures=[], seed=seed,
                        score_cells_replayed=cells, queries_replayed=4*cfg['fixture_queries_per_policy'],
                        cpu_gpu_score_bits_exact=True, source_summary_sha256=sha(out/'summary.json'),
                        fixture_sha256=sha(out/'fixtures.npz'), controls=controls(),
                        limits=result['limits'])
    assert not (out/'audit.json').exists()
    (out/'audit.json').write_text(json.dumps(audit_result, indent=2))
    print('RANK_CONFLICT_AUDIT_PASS', seed, cells, flush=True)


def coordinate(root, cfg, config):
    canonical = root/'experiments'/cfg['id']/'experiment.yaml'
    state = yaml.safe_load(canonical.read_text())
    assert state['status'] == 'registered'
    for name, digest in state['frozen_input_sha256'].items():
        assert sha(root/name) == digest, name
    for dependency in ('KGE-ENTITY-COMMON-RANK-CAPACITY-037', 'KGE-ENTITY-COMMON-RANK-CAPACITY-UTILITY-038'):
        previous = yaml.safe_load((root/'experiments'/dependency/'experiment.yaml').read_text())
        assert not previous['active_processes'] and 'terminal' in previous['status']
        assert all(v['status'] == 'completed_audited' for v in previous['jobs'].values())
    state.update(status='running_train_only_diagnostic', coordinator_pid=os.getpid(), jobs={}, active_processes=[])
    active = {}
    def save():
        state['active_processes']=[dict(seed=seed, pid=row['proc'].pid, stage=row['stage'], gpu=row['gpu']) for seed,row in active.items()]
        tmp = canonical.with_suffix('.tmp')
        tmp.write_text(yaml.safe_dump(state, sort_keys=False))
        tmp.replace(canonical)
    def start(seed, stage, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if stage == 'audit' else str(gpu),
                   OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        log = (root/cfg['log_directory']/f's{seed}-{stage}.log').open('xb')
        cmd=[os.path.abspath(root/cfg['python']), '-u', str(Path(__file__)), '--root', str(root), '--config', config, '--stage', stage, '--seed', str(seed)]
        process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[seed]=dict(proc=process,log=log,stage=stage,gpu=gpu)
        state['jobs'][str(seed)]=dict(status='running',stage=stage,pid=process.pid,gpu=gpu)
        print('CONFLICT_START',seed,stage,process.pid,gpu,flush=True)
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle={int(i) for line in usage.splitlines() for i,m,u in [line.split(',')] if int(m)<=1024 and int(u)<=20}
    assert set(cfg['gpu_indices']) <= idle
    for seed,gpu in zip(cfg['seeds'],cfg['gpu_indices']):
        start(seed,'run',gpu)
    save()
    while active:
        for seed, row in list(active.items()):
            rc=row['proc'].poll()
            if rc is None:
                continue
            row['log'].close()
            del active[seed]
            if rc == 0 and row['stage'] == 'run':
                start(seed,'audit',None)
            else:
                state['jobs'][str(seed)]=dict(status='completed_audited' if rc == 0 else 'failed_retained',stage=row['stage'],exit_code=rc)
                print('CONFLICT_TERMINAL',seed,row['stage'],rc,flush=True)
        save()
        if active:
            time.sleep(3)
    dest=root/'results'/cfg['id']
    dest.mkdir(exist_ok=False)
    rows=[]
    for seed in cfg['seeds']:
        if state['jobs'][str(seed)]['status']=='completed_audited':
            folder=root/cfg['output_directory']/f's{seed}'
            rows.append(dict(seed=seed, result=json.loads((folder/'summary.json').read_text()), audit=json.loads((folder/'audit.json').read_text())))
    (dest/'summary.json').write_text(json.dumps(dict(scope=cfg['scope'],rows=rows,total_seeds=3,audited_seeds=len(rows),scientific_utility_asserted=False),indent=2))
    state.update(status='completed_audited_train_diagnostic_only' if len(rows)==3 else 'terminal_with_retained_failures',
                 result_directory=str(dest.relative_to(root)),completed_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',default='configs/rank-conflict39.json')
    parser.add_argument('--stage',choices=['run','audit','coordinate'],required=True)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    root=args.root.resolve()
    cfg=json.loads((root/args.config).read_text())
    if args.stage=='coordinate':
        coordinate(root,cfg,args.config)
    elif args.stage=='run':
        worker(root,cfg,args.seed)
    else:
        audit(root,cfg,args.seed)
