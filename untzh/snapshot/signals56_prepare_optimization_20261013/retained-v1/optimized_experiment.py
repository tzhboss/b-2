"""Direct experiment: prepare -> calibrate/train -> evaluate -> summarize."""
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from fast_adapter import PrefixStructure, stream
from calibration_parts import build_scales, coefficient

ROOT = Path(__file__).resolve().parent
SRC = Path('/data/lc/tzh/experiment4')
N, TOTAL = 7505, 11258
ARMS = ['absolute', 'relative', 'random', 'structural']
SEED = 550017
DEVICE = 'cuda'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def key(row):
    h, r, _ = map(int, row)
    return h, r // 2, 'head' if r % 2 else 'tail'


def aggregate(scores, gold, scale, mask):
    z = (5 * F.relu((scores - gold[..., None]) / scale[..., None])).masked_fill(~mask, -torch.inf)
    shift = z.max(-1).values.detach()
    return (shift + torch.logsumexp(z - shift[..., None], -1) - mask.sum(-1).double().log()) / 5


def penalty(E, cache, arm):
    scores = cache['q'] @ E[N:].double().T
    mask = cache['mask']
    if arm == 'absolute':
        each = F.softplus(scores).masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
    else:
        u = aggregate(scores, cache['gold'], cache['scale'], mask)
        each = u.square() if arm == 'relative' else F.relu(u - cache[arm + '_mean']).square()
    return each.mean()


def learning(E, R, new, tape, step):
    lo, hi = tape['bounds'][step:step + 2]
    p, nh, nt = [torch.as_tensor(x, device=DEVICE, dtype=torch.long) for x in
                 (new[tape['pos'][lo:hi]], tape['nh'][lo:hi], tape['nt'][lo:hi])]
    h, r, t = p.unbind(1)
    ps = (E[h] * R[r] * E[t]).sum(-1)
    ns = (E[nh] * R[r[:, None]] * E[nt]).sum(-1)
    return F.softplus(-ps).mean() + F.softplus(ns).mean() + .0005 * (E[h].square().mean() + R[r].square().mean())


def prepare(initial, old, new, tape):
    start = time.monotonic()
    cache_dir = ROOT / 'cache'
    cache_dir.mkdir(exist_ok=False)
    facts = {(int(h), int(r) // 2, int(t)) if r % 2 == 0 else (int(t), int(r) // 2, int(h))
             for h, r, t in np.concatenate([old, new])}
    structure = PrefixStructure(facts, range(N), range(N, TOTAL))
    eligible = np.flatnonzero(load(SRC / 'full_increment_screen49_20261012/artifacts/evaluation/A.npz')['old_train'][:, 1] <= 10)
    rng = np.random.default_rng(SEED)
    cycle, cursor, sequence = rng.permutation(eligible), 0, []
    for _ in tape['epochs']:
        ids = []
        while len(ids) < 32:
            if cursor == len(cycle):
                cycle, cursor = rng.permutation(eligible), 0
            take = min(32 - len(ids), len(cycle) - cursor)
            ids.extend(cycle[cursor:cursor + take])
            cursor += take
        sequence.append(ids)
    sequence = np.asarray(sequence, dtype=np.int32)
    np.savez_compressed(ROOT / 'protection-sequence.npz', old_ids=sequence)
    e, r = initial['E'].double().numpy(), initial['R'].double().numpy()
    scale_records = {}
    times = collections.defaultdict(float)
    tick = time.monotonic()
    for row in old[sequence].reshape(-1, 3):
        k = key(row)
        if k in scale_records:
            continue
        known = structure.truth[k]
        pool = [i for i in structure.old if i not in known]
        if not pool or not any(i not in known for i in structure.new):
            continue
        ids = stream(SEED, list(k), 'scale').choice(pool, size=min(256, len(pool)), replace=False)
        scale_records[k] = (e[row[0]] * r[row[1]]) @ e[ids].T
    scales = build_scales(scale_records.items(), eta=1.4826, epsilon=1e-8, adequate=32)
    times['scale'] = time.monotonic() - tick
    scale_summary = np.quantile([v['scale'] for v in scales.values()], [0, .25, .5, .75, 1]).tolist()
    del scale_records
    oldE, R = initial['E'][:N].double(), initial['R'].double()
    kept = 0
    for j, ids in enumerate(sequence):
        tick = time.monotonic()
        rows, masks, random_refs, structural_refs = [], [], [], []
        for row in old[ids]:
            k = key(row)
            cohorts = structure.cohorts(k, int(row[2]), seed=SEED, draws=4, scale_count=256)
            if not cohorts['eligible']:
                continue
            cols = np.asarray(cohorts['new']) - N
            mask = np.zeros(TOTAL - N, dtype=bool)
            mask[cols] = True
            sr = np.zeros((4, TOTAL - N), dtype=np.int64)
            rr = np.zeros_like(sr)
            sr[:, cols] = np.asarray(cohorts['references'])
            for draw in range(4):
                rand = stream(SEED, list(k), 'random_reference', draw)
                values = list(map(int, rand.permutation(cohorts['old_pool'])))[:len(cols)]
                if len(values) < len(cols):
                    values += list(map(int, rand.choice(cohorts['old_pool'], len(cols) - len(values), replace=True)))
                rr[draw, cols] = values
            rows.append(row)
            masks.append(mask)
            random_refs.append(rr)
            structural_refs.append(sr)
        times['reference_matching'] += time.monotonic() - tick
        if not rows:
            raise ValueError('empty protection batch')
        tick = time.monotonic()
        queries = np.asarray(rows)
        mask = torch.from_numpy(np.asarray(masks))
        q = oldE[queries[:, 0]] * R[queries[:, 1]]
        gold = (q * oldE[queries[:, 2]]).sum(-1)
        tau = torch.tensor([scales[key(row)]['scale'] for row in queries], dtype=torch.float64)
        scores = q @ oldE.T
        payload = {'q': q.numpy(), 'gold': gold.numpy(), 'scale': tau.numpy(), 'mask': mask.numpy()}
        for arm, values in [('random', random_refs), ('structural', structural_refs)]:
            ref = torch.from_numpy(np.asarray(values))
            rm = mask[:, None, :].expand_as(ref)
            ref_scores = torch.gather(scores[:, None, :].expand(-1, 4, -1), 2, ref)
            payload[arm + '_mean'] = aggregate(ref_scores, gold[:, None], tau[:, None], rm).mean(-1).numpy()
        times['reference_aggregation'] += time.monotonic() - tick
        tick = time.monotonic()
        np.savez_compressed(cache_dir / f'{j:04d}.npz', **payload)
        times['cache_io'] += time.monotonic() - tick
        kept += len(rows)
        if j % 32 == 0:
            print('PREPARE', j, len(sequence), round(time.monotonic() - start, 2), flush=True)
            write(ROOT / 'status.json', {'stage': 'prepare', 'step': j, 'total': len(sequence), 'seconds': time.monotonic() - start})
    report = {'seconds': time.monotonic() - start, 'components': dict(times), 'steps': len(sequence),
              'eligible_old_answers': len(eligible), 'kept_occurrences': kept, 'scale_quantiles': scale_summary,
              'cache_bytes': sum(p.stat().st_size for p in cache_dir.glob('*.npz'))}
    write(ROOT / 'prepare.json', report)
    return report


def cache(step):
    return {k: torch.as_tensor(v, device=DEVICE) for k, v in load(ROOT / 'cache' / f'{step:04d}.npz').items()}


def calibrate(initial, new, tape):
    start = time.monotonic()
    E = torch.nn.Parameter(initial['E'].to(DEVICE).clone())
    R = initial['R'].to(DEVICE).clone()
    base_norms = [float(torch.linalg.vector_norm(torch.autograd.grad(learning(E, R, new, tape, j), E)[0][N:].double())) for j in range(8)]
    result = {}
    for arm in ARMS:
        norms = [float(torch.linalg.vector_norm(torch.autograd.grad(penalty(E, cache(j), arm), E)[0][N:].double())) for j in range(8)]
        result[arm] = {**coefficient(base_norms, norms, rho=.1), 'base_norms': base_norms, 'protection_norms': norms}
        print('COEFFICIENT', arm, result[arm]['lambda'], flush=True)
    write(ROOT / 'coefficients.json', result)
    return result, time.monotonic() - start


def train(initial, new, tape, arm, coef):
    start = time.monotonic()
    out = ROOT / 'models' / arm
    out.mkdir(parents=True, exist_ok=False)
    E = torch.nn.Parameter(initial['E'].to(DEVICE).clone())
    R = initial['R'].to(DEVICE).clone()
    E.register_hook(lambda g: torch.cat((torch.zeros_like(g[:N]), g[N:])))
    optimizer = torch.optim.Adam([E], lr=.007)
    steps = []
    for j, epoch in enumerate(tape['epochs']):
        base = learning(E, R, new, tape, j)
        protect = penalty(E, cache(j), arm)
        loss = base + coef * protect
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(E.grad).all():
            raise FloatingPointError('nonfinite training gradient')
        optimizer.step()
        steps.append({'step': j, 'epoch': int(epoch), 'learning_loss': float(base.detach()), 'protection_loss': float(protect.detach()), 'loss': float(loss.detach())})
        if j == len(tape['epochs']) - 1 or tape['epochs'][j + 1] != epoch:
            print('TRAIN', arm, int(epoch), steps[-1]['loss'], flush=True)
            write(ROOT / 'status.json', {'stage': 'train', 'arm': arm, 'epoch': int(epoch), 'seconds': time.monotonic() - start})
    torch.save({'E': E.detach().cpu().double(), 'R': R.cpu().double()}, out / 'epoch-20.pt')
    write(out / 'losses.json', steps)
    result = {'seconds': time.monotonic() - start, 'steps': len(steps), 'coefficient': coef,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated()}
    write(out / 'training.json', result)
    return result


@torch.no_grad()
def ranks(E, R, queries, truths, arm=None, split=None):
    result = []
    for i in range(0, len(queries), 64):
        a = queries[i:i + 64]
        q = torch.as_tensor(a, device=DEVICE)
        scores = (E[q[:, 0]] * R[q[:, 1]]) @ E.T
        gold = scores[torch.arange(len(q), device=DEVICE), q[:, 2]].clone()
        raw = 1 + (scores > gold[:, None]).sum(1)
        rows, cols = [], []
        for j, (h, r, t) in enumerate(a):
            for candidate in truths[int(h), int(r)]:
                if candidate != t and candidate < len(E):
                    rows.append(j)
                    cols.append(candidate)
        if rows:
            scores[rows, cols] = -torch.inf
        filtered = 1 + (scores > gold[:, None]).sum(1)
        result.append(torch.stack([raw, filtered], 1).cpu().numpy())
        if i % 4096 == 0:
            write(ROOT / 'status.json', {'stage': 'evaluate', 'arm': arm, 'split': split,
                                        'completed_queries': min(i + len(a), len(queries)), 'total_queries': len(queries)})
    return np.concatenate(result)


def metrics(a, queries):
    masks = {'all': np.ones(len(queries), bool),
             'head_old_gold_new': (queries[:, 0] < N) & (queries[:, 2] >= N),
             'head_new_gold_old': (queries[:, 0] >= N) & (queries[:, 2] < N),
             'both_new': (queries[:, 0] >= N) & (queries[:, 2] >= N)}
    result = {}
    for direction, mask in masks.items():
        if not mask.any():
            continue
        result[direction] = {}
        for j, name in enumerate(['raw', 'filtered_train_truths']):
            r = a[mask, j]
            result[direction][name] = {'n': len(r), 'mrr': float(np.mean(1 / r)), 'hits1': float(np.mean(r <= 1)), 'hits10': float(np.mean(r <= 10))}
    return result


def evaluate(initial, old, new):
    start = time.monotonic()
    out = ROOT / 'evaluation'
    out.mkdir(exist_ok=False)
    truths = collections.defaultdict(set)
    for h, r, t in np.concatenate([old, new]):
        truths[int(h), int(r)].add(int(t))
    valid = load(SRC / 'entity_refresh43_20261012/artifacts/data/validation.npz')
    queries = {'old_train': old, 'old_valid': valid['old'], 'new_valid': valid['new']}
    np.savez_compressed(out / 'queries.npz', **queries)
    # Reuse the exact existing ordinary-learning and replay runs, not new repetitions.
    prior = SRC / 'replay_strength53_20261013/artifacts/evaluation'
    paths = {'new_only': prior / 'new_only-e20.npz', 'replay_010': prior / 'replay_010-e20.npz',
             'replay_030': prior / 'replay_030-e20.npz', 'replay_100': prior / 'absolute_replay-e20.npz'}
    states = {name: load(path) for name, path in paths.items()}
    A = load(prior / 'A.npz')
    report = {}
    for arm in ARMS:
        model = torch.load(ROOT / 'models' / arm / 'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = model['E'].to(DEVICE), model['R'].to(DEVICE)
        state = {}
        for split, q in queries.items():
            state[split] = ranks(E, R, q, truths, arm=arm, split=split)
        np.savez_compressed(out / f'{arm}.npz', **state)
        states[arm] = state
        print('EVALUATE', arm, round(time.monotonic() - start, 2), flush=True)
        write(ROOT / 'status.json', {'stage': 'evaluate', 'arm': arm, 'seconds': time.monotonic() - start})
    for name, state in states.items():
        report[name] = {split: metrics(state[split], q) for split, q in queries.items()}
        for split in ['old_train', 'old_valid']:
            drops = {}
            for j, kind in enumerate(['raw', 'filtered_train_truths']):
                eligible = A[split][:, j] <= 10
                drops[kind] = {'eligible': int(eligible.sum()), 'drops': int((eligible & (state[split][:, j] > 10)).sum())}
            report[name][split]['drops'] = drops
    write(out / 'metrics.json', report)
    write(out / 'reused_controls.json', {k: str(v) for k, v in paths.items()})
    return report, time.monotonic() - start


def summarize(report, costs):
    lines = ['# Protection signal comparison 056', '',
             'Seed17, same initial state and learning tape, 20 epochs. Old entity/relation rows fixed. '
             'New arms use 32 filtered-qualified TRAIN answers per step, complete new candidate pool, '
             'and initial 8-batch gradient RMS ratio 0.1. Reference arms are explicit MEOR adaptations, not author-native reproductions. '
             'Replay controls are reused 052/053 runs with their original sampling/strength; they are contextual tradeoff points, not identically calibrated signal controls.', '',
             'Filtering uses TRAIN truths only. VALID is an exposed development set. No TEST evaluation or independent repetitions.', '',
             '| Arm | New raw MRR | New filtered MRR | Old head → new gold filtered MRR ratio | New head → old gold ratio | Both new ratio | Old TRAIN filtered drops | Old VALID filtered drops | Old VALID filtered MRR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    baseline = report['new_only']['new_valid']
    for name, m in report.items():
        nv = m['new_valid']
        ratio = [nv[d]['filtered_train_truths']['mrr'] / baseline[d]['filtered_train_truths']['mrr'] for d in ['head_old_gold_new', 'head_new_gold_old', 'both_new']]
        lines.append(f"| {name} | {nv['all']['raw']['mrr']:.9f} | {nv['all']['filtered_train_truths']['mrr']:.9f} | {ratio[0]:.6f} | {ratio[1]:.6f} | {ratio[2]:.6f} | {m['old_train']['drops']['filtered_train_truths']['drops']} | {m['old_valid']['drops']['filtered_train_truths']['drops']} | {m['old_valid']['all']['filtered_train_truths']['mrr']:.9f} |")
    lines += ['', 'Timing records include actual shared reference preparation and initial calibration. '
              'Preparation computes both reference arms, so combined preparation is not a per-method deployment cost. '
              'Learning tape and controls were reused; their historical generation/training costs are not zero.', '',
              'Full raw/filtered directional MRR, Hits1, Hits10, old MRR and fixed-eligibility drops are in evaluation/metrics.json.', '',
              'No CPU audit, hash/freeze/closure gates, synthetic qualification loops, or archive stage were executed.']
    (ROOT / 'results.md').write_text('\n'.join(lines) + '\n')
    write(ROOT / 'costs.json', costs)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    write(ROOT / 'status.json', {'stage': 'starting', 'pid': os.getpid()})
    initial = torch.load(SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt', map_location='cpu', weights_only=True)
    d = load(SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old, new = d['old'], d['new']
    tape = load(SRC / 'matched_protection52_20261013/artifacts/tape/learning.npz')
    print('INPUT', initial['E'].dtype, tuple(initial['E'].shape), len(tape['epochs']), flush=True)
    costs = {'prepare': prepare(initial, old, new, tape)}
    write(ROOT / 'status.json', {'stage': 'calibrate'})
    coefficients, seconds = calibrate(initial, new, tape)
    costs['calibration_seconds'] = seconds
    costs['train'] = {}
    for arm in ARMS:
        costs['train'][arm] = train(initial, new, tape, arm, coefficients[arm]['lambda'])
    report, seconds = evaluate(initial, old, new)
    costs['evaluation_seconds'] = seconds
    costs['total_seconds'] = time.monotonic() - started
    summarize(report, costs)
    write(ROOT / 'status.json', {'stage': 'complete', 'seconds': time.monotonic() - started, 'pid': os.getpid()})
    print('COMPLETE', round(time.monotonic() - started, 2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(ROOT / 'status.json', {'stage': 'failed', 'error': repr(exc), 'pid': os.getpid()})
        raise
