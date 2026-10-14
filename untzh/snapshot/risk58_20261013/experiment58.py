"""Cached seed17 risk forecast -> matched simple protection experiment."""
import collections
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SRC = Path('/data/lc/tzh/experiment4')
SOURCE56 = Path('/data/lc/untzh/signals56_20261013')
sys.path.insert(0, str(SOURCE56))
import experiment as base
base.ROOT = ROOT
N, TOTAL, SIZE, SAMPLES, SEED = 7505, 11258, 16384, 128, 17
ARMS = ['uniform', 'boundary', 'pressure32', 'pressure128', 'lookahead', 'logistic', 'tree']
FEATURE_NAMES = ['old_rank', 'capacity', 'gold_over_tau', 'qnorm', 'head_train_count',
                 'relation_train_count', 'gold_train_count', 'known_new_count',
                 'max_gap32', 'max_gap128', 'top4_gap', 'mean_gap', 'std_gap',
                 'fraction_above', 'mean_softplus_gap', 'proposal_max_gap',
                 'extrapolated_max_gap', 'extrapolated_fraction_above',
                 'max_delta', 'mean_abs_delta', 'epoch_fraction']


def dump(name, value):
    base.write(ROOT / name, value)


def sync():
    torch.cuda.synchronize()


def progress(stage, **extra):
    dump('status.json', {'stage': stage, 'pid': os.getpid(), **extra})
    print(stage.upper(), extra, flush=True)


def init_model(initial):
    E = torch.nn.Parameter(initial['E'].cuda().clone())
    R = initial['R'].cuda().clone()
    E.register_hook(lambda g: torch.cat((torch.zeros_like(g[:N]), g[N:])))
    return E, R, torch.optim.Adam([E], lr=.007)


def gradient(E, R, new, tape, step):
    loss = base.learning(E, R, new, tape, step)
    return loss, torch.autograd.grad(loss, E)[0]


@torch.no_grad()
def proposal(E, optimizer, g):
    """Exact PyTorch Adam proposal on copied state; host state remains untouched."""
    trial = torch.nn.Parameter(E.detach().clone())
    opt = torch.optim.Adam([trial], lr=.007)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    trial.grad = g.detach().clone()
    opt.step()
    return trial.detach() - E.detach()


def cohort(initial, old, new, A):
    start = time.monotonic()
    path = ROOT / 'cache/cohort.npz'
    truth = collections.defaultdict(set)
    for h, r, t in np.concatenate([old, new]):
        truth[int(h), int(r)].add(int(t))
    if path.exists():
        payload = base.load(path)
    else:
        rng = np.random.default_rng(SEED)
        ids = rng.choice(np.flatnonzero(A['old_train'][:, 1] <= 10), SIZE, replace=False)
        queries = old[ids]
        groups = sorted({(int(h), int(r)) for h, r, t in queries})
        group_order = rng.permutation(len(groups))
        mapping = {groups[int(j)]: (0 if i < .6 * len(groups) else 1 if i < .8 * len(groups) else 2)
                   for i, j in enumerate(group_order)}
        split = np.array([mapping[int(h), int(r)] for h, r, t in queries], dtype=np.int8)
        candidates = rng.integers(N, TOTAL, size=(SIZE, SAMPLES), dtype=np.int64)
        old_sample = rng.integers(0, N, size=(SIZE, 64), dtype=np.int64)
        known_counts = []
        for i, (h, r, t) in enumerate(queries):
            known = truth[int(h), int(r)]
            bad = np.isin(candidates[i], list(known))
            while bad.any():
                candidates[i, bad] = rng.integers(N, TOTAL, size=int(bad.sum()))
                bad = np.isin(candidates[i], list(known))
            known_counts.append(sum(t >= N for t in known))
        e, rel = initial['E'].double(), initial['R'].double()
        q = e[queries[:, 0]] * rel[queries[:, 1]]
        gold = (q * e[queries[:, 2]]).sum(1)
        tau = torch.empty(SIZE, dtype=torch.float64)
        for lo in range(0, SIZE, 256):
            hi = min(lo + 256, SIZE)
            tau[lo:hi] = (q[lo:hi, None] * e[old_sample[lo:hi]]).sum(-1).std(-1).clamp_min(1e-8)
        head_count = np.bincount(np.concatenate([old, new])[:, 0], minlength=TOTAL)
        tail_count = np.bincount(np.concatenate([old, new])[:, 2], minlength=TOTAL)
        rel_count = np.bincount(np.concatenate([old, new])[:, 1], minlength=len(rel))
        oldrank = A['old_train'][ids, 1]
        fixed = np.column_stack([oldrank, 11 - oldrank, gold.numpy() / tau.numpy(),
                                 q.norm(dim=1).numpy(), np.log1p(head_count[queries[:, 0]]),
                                 np.log1p(rel_count[queries[:, 1]]), np.log1p(tail_count[queries[:, 2]]),
                                 known_counts]).astype(np.float32)
        payload = dict(ids=ids, queries=queries, split=split, candidates=candidates, q=q.numpy(),
                       gold=gold.numpy(), tau=tau.numpy(), oldrank=oldrank, fixed=fixed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **payload)
    known_rows, known_cols = [], []
    for i, (h, r, t) in enumerate(payload['queries']):
        for candidate in truth[int(h), int(r)]:
            if candidate >= N:
                known_rows.append(i)
                known_cols.append(candidate - N)
    payload['known_rows'] = np.asarray(known_rows, dtype=np.int64)
    payload['known_cols'] = np.asarray(known_cols, dtype=np.int64)
    dump('cohort.json', {'seconds': time.monotonic() - start, 'answers': SIZE,
                         'population_eligible': int((A['old_train'][:, 1] <= 10).sum()),
                         'query_groups': len(set(map(tuple, payload['queries'][:, :2]))),
                         'split_answers': np.bincount(payload['split'], minlength=3).tolist(),
                         'sampled_candidates': SAMPLES, 'features': FEATURE_NAMES})
    return payload, truth


@torch.no_grad()
def features(E, delta, data, epoch):
    """No full-pool ranks or future labels are used here."""
    sync(); start = time.monotonic()
    e = E.detach().double()
    de = delta.double()
    blocks = []
    for lo in range(0, SIZE, 256):
        hi = min(lo + 256, SIZE)
        ids = torch.as_tensor(data['candidates'][lo:hi], device='cuda')
        q = torch.as_tensor(data['q'][lo:hi], device='cuda')
        gold = torch.as_tensor(data['gold'][lo:hi], device='cuda')
        tau = torch.as_tensor(data['tau'][lo:hi], device='cuda')
        z = ((q[:, None] * e[ids]).sum(-1) - gold[:, None]) / tau[:, None]
        dz = (q[:, None] * de[ids]).sum(-1) / tau[:, None]
        f = z + 37 * dz
        values = torch.stack([z[:, :32].max(-1).values, z.max(-1).values,
                              z.topk(4, dim=-1).values.mean(-1), z.mean(-1), z.std(-1),
                              (z > 0).double().mean(-1), F.softplus(z).mean(-1),
                              (z + dz).max(-1).values, f.max(-1).values,
                              (f > 0).double().mean(-1), dz.max(-1).values,
                              dz.abs().mean(-1), torch.full_like(gold, epoch / 20)], 1)
        blocks.append(values.float().cpu().numpy())
    X = np.column_stack([data['fixed'], np.concatenate(blocks)])
    assert X.shape == (SIZE, len(FEATURE_NAMES)) and np.isfinite(X).all()
    sync()
    return X, time.monotonic() - start


@torch.no_grad()
def full_ranks(E, data):
    sync(); start = time.monotonic()
    enew = E.detach()[N:].double()
    counts = []
    for lo in range(0, SIZE, 256):
        hi = min(lo + 256, SIZE)
        q = torch.as_tensor(data['q'][lo:hi], device='cuda')
        gold = torch.as_tensor(data['gold'][lo:hi], device='cuda')
        above = q @ enew.T > gold[:, None]
        mask = (data['known_rows'] >= lo) & (data['known_rows'] < hi)
        above[data['known_rows'][mask] - lo, data['known_cols'][mask]] = False
        counts.append(above.sum(1).cpu().numpy())
    sync()
    return data['oldrank'] + np.concatenate(counts), time.monotonic() - start


def make_trajectory(initial, old, new, tape, data):
    marker = ROOT / 'cache/trajectory.json'
    if marker.exists():
        return json.loads(marker.read_text())
    E, R, opt = init_model(initial)
    starts = [int(np.flatnonzero(tape['epochs'] == ep)[0]) for ep in range(1, 21)]
    costs = {'features_seconds': 0., 'full_label_scan_seconds': 0., 'host_training_seconds': 0.}
    for epoch in range(21):
        ranks, seconds = full_ranks(E, data)
        costs['full_label_scan_seconds'] += seconds
        if epoch < 20:
            step = starts[epoch]
            _, g = gradient(E, R, new, tape, step)
            delta = proposal(E, opt, g)
            X, seconds = features(E, delta, data, epoch)
            costs['features_seconds'] += seconds
            np.savez(ROOT / 'cache' / f'frame-{epoch:02d}.npz', X=X, ranks=ranks)
        else:
            np.savez(ROOT / 'cache/frame-20.npz', ranks=ranks)
        if epoch == 10:
            torch.save({'E': E.detach().cpu(), 'R': R.cpu(), 'optimizer': opt.state_dict()}, ROOT / 'cache/prefix10.pt')
        if epoch < 20:
            start = time.monotonic()
            end = starts[epoch + 1] if epoch < 19 else len(tape['epochs'])
            for j in range(starts[epoch], end):
                loss, g = gradient(E, R, new, tape, j)
                if not torch.isfinite(loss) or not torch.isfinite(g).all():
                    raise FloatingPointError('nonfinite base trajectory')
                opt.zero_grad(set_to_none=True)
                E.grad = g
                opt.step()
            sync(); costs['host_training_seconds'] += time.monotonic() - start
        progress('trajectory', epoch=epoch, dropped=int((ranks > 10).sum()))
    historical = torch.load(SRC / 'matched_protection52_20261013/artifacts/train/new_only/epoch-20.pt',
                            map_location='cpu', weights_only=True)
    equal = torch.equal(E.detach().cpu().double(), historical['E']) and torch.equal(R.cpu().double(), historical['R'])
    if not equal:
        raise RuntimeError('ordinary trajectory differs from reused baseline')
    costs['historical_endpoint_equal'] = equal
    dump('cache/trajectory.json', costs)
    return costs


def rule_scores(X):
    logcap = np.log(X[:, 1])
    return {'boundary': X[:, 0], 'pressure32': X[:, 8] - logcap,
            'pressure128': X[:, 9] - logcap, 'lookahead': X[:, 16] - logcap}


def selection_metrics(y, scores, eligible=None):
    if eligible is None:
        eligible = np.ones(len(y), bool)
    y, scores = y[eligible].astype(bool), scores[eligible]
    result = {'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()) if len(y) else None}
    if not len(y):
        return result
    result['average_precision'] = float(average_precision_score(y, scores)) if y.any() else None
    result['roc_auc'] = float(roc_auc_score(y, scores)) if y.any() and not y.all() else None
    order = np.argsort(-scores, kind='stable')
    for fraction in [.05, .1, .2]:
        k = max(1, int(np.ceil(len(y) * fraction)))
        hits = int(y[order[:k]].sum())
        result[str(fraction)] = {'selected': k, 'hits': hits, 'precision': hits / k,
                                 'recall': hits / int(y.sum()) if y.any() else None}
    return result


def fit_predictors(data):
    frames = [base.load(ROOT / 'cache' / f'frame-{i:02d}.npz') for i in range(21)]
    trainmask = data['split'] == 0
    X = np.concatenate([frames[i]['X'][trainmask] for i in range(8)])
    y = np.concatenate([(frames[i + 1]['ranks'][trainmask] > 10).astype(int) for i in range(8)])
    if y.sum() < 10 or (1 - y).sum() < 10:
        raise RuntimeError('insufficient risk events for fixed learners')
    start = time.monotonic()
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=300, random_state=SEED))
    logistic.fit(X, y)
    ltime = time.monotonic() - start
    start = time.monotonic()
    weights = np.where(y, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))
    tree = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, max_depth=3,
                                          learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=SEED)
    tree.fit(X, y, sample_weight=weights)
    ttime = time.monotonic() - start
    models = {'logistic': logistic, 'tree': tree}
    for name, model in models.items():
        joblib.dump(model, ROOT / 'cache' / f'{name}.joblib')
    rng = np.random.default_rng(SEED)
    report = {'fit_samples': len(y), 'fit_positive_events': int(y.sum()),
              'fit_seconds': {'logistic': ltime, 'tree': ttime}, 'groups': {}}
    for group, epochs, split in [('development', range(8, 10), 1), ('late_heldout', range(10, 20), 2)]:
        mask = data['split'] == split
        rows = collections.defaultdict(list)
        targets, safe = [], []
        for i in epochs:
            Xi = frames[i]['X'][mask]
            scores = rule_scores(Xi)
            scores['random'] = rng.random(len(Xi))
            for name, model in models.items():
                scores[name] = model.predict_proba(Xi)[:, 1]
            for name, score in scores.items():
                rows[name].append(score)
            targets.append(frames[i + 1]['ranks'][mask] > 10)
            safe.append(frames[i]['ranks'][mask] <= 10)
        target, safe = np.concatenate(targets), np.concatenate(safe)
        report['groups'][group] = {name: {'all_next_drop': selection_metrics(target, np.concatenate(values)),
                                         'new_drop_among_current_safe': selection_metrics(target, np.concatenate(values), safe)}
                                   for name, values in rows.items()}
    dump('risk-prediction.json', report)
    progress('risk_fit', fit_positive_events=int(y.sum()), seconds=ltime + ttime)
    return models, report


def protection(E, data, selected):
    ids = torch.as_tensor(data['candidates'][selected], device='cuda')
    q = torch.as_tensor(data['q'][selected], device='cuda')
    scores = (q[:, None] * E[ids].double()).sum(-1)
    return F.softplus(scores).mean()


def calibrate(prefix, initial, new, tape, data):
    E, R, _ = init_model(initial)
    E.data.copy_(prefix['E'].cuda())
    rng = np.random.default_rng(SEED)
    bn, pn = [], []
    for step in range(370, 378):
        _, g = gradient(E, R, new, tape, step)
        p = torch.autograd.grad(protection(E, data, rng.choice(SIZE, 32, replace=False)), E)[0]
        bn.append(float(g[N:].double().norm()))
        pn.append(float(p[N:].double().norm()))
    coefficient = .1 * np.sqrt(np.mean(np.square(bn))) / np.sqrt(np.mean(np.square(pn)))
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise FloatingPointError('bad shared calibration')
    dump('coefficient.json', {'lambda': coefficient, 'rho': .1, 'base_norms': bn, 'protection_norms': pn,
                             'rule': 'shared uniform prefix10 calibration; stepwise cap for every arm'})
    return coefficient


def train_arm(arm, initial, prefix, new, tape, data, models, coefficient):
    start = time.monotonic()
    E, R, opt = init_model(initial)
    E.data.copy_(prefix['E'].cuda())
    opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    rng = np.random.default_rng(SEED)
    records, routing, selection_seconds = [], [], 0.
    top = np.arange(SIZE)
    for j in range(370, 740):
        epoch = int(tape['epochs'][j])
        if j == 370 or tape['epochs'][j - 1] != epoch:
            _, g0 = gradient(E, R, new, tape, j)
            tick = time.monotonic()
            if arm == 'uniform':
                scores = np.zeros(SIZE)
            elif arm == 'boundary':
                scores = data['oldrank'].astype(float)
            else:
                delta = proposal(E, opt, g0) if arm in ['lookahead', 'logistic', 'tree'] else torch.zeros_like(E)
                X, _ = features(E, delta, data, epoch - 1)
                scores = models[arm].predict_proba(X)[:, 1] if arm in models else rule_scores(X)[arm]
            top = np.argsort(-scores, kind='stable')[:int(np.ceil(.2 * SIZE))]
            sync(); spent = time.monotonic() - tick
            selection_seconds += spent
            routing.append({'epoch': epoch, 'seconds': spent, 'pool_answers': len(top),
                            'pool_query_groups': len(set(map(tuple, data['queries'][top, :2])))})
        selected = rng.choice(SIZE, 32, replace=False) if arm == 'uniform' else np.concatenate([
            rng.choice(top, 26, replace=False), rng.choice(SIZE, 6, replace=False)])
        base_loss, gl = gradient(E, R, new, tape, j)
        pen = protection(E, data, selected)
        gp = torch.autograd.grad(pen, E)[0]
        bn, pn = float(gl[N:].double().norm()), float(gp[N:].double().norm())
        alpha = min(coefficient, .1 * bn / pn) if pn > 0 else coefficient
        g = gl + alpha * gp
        if not torch.isfinite(base_loss) or not torch.isfinite(pen) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite risk-selected protection')
        opt.zero_grad(set_to_none=True); E.grad = g; opt.step()
        records.append({'step': j, 'epoch': epoch, 'learning_loss': float(base_loss), 'protection_loss': float(pen),
                        'alpha': alpha, 'gradient_ratio': alpha * pn / bn if bn else 0.})
        if j == 739 or tape['epochs'][j + 1] != epoch:
            progress('train', arm=arm, epoch=epoch, seconds=time.monotonic() - start)
    assert torch.equal(E[:N].detach().cpu(), initial['E'][:N]) and torch.equal(R.cpu(), initial['R'])
    out = ROOT / 'models' / arm; out.mkdir(parents=True, exist_ok=True)
    torch.save({'E': E.detach().cpu().double(), 'R': R.cpu().double()}, out / 'epoch-20.pt')
    base.write(out / 'steps.json', records)
    base.write(out / 'routing.json', routing)
    result = {'seconds': time.monotonic() - start, 'selection_seconds_included': selection_seconds,
              'steps': 370, 'warmup_epochs': 10, 'protection_epochs': 10,
              'old_rows_frozen': True, 'max_gradient_ratio': max(r['gradient_ratio'] for r in records)}
    base.write(out / 'training.json', result)
    return result


def evaluate(old, new, truth, A):
    started = time.monotonic()
    valid = base.load(SRC / 'entity_refresh43_20261012/artifacts/data/validation.npz')
    queries = {'old_train': old, 'old_valid': valid['old'], 'new_valid': valid['new']}
    out = ROOT / 'evaluation'; out.mkdir(exist_ok=True)
    prior = SRC / 'replay_strength53_20261013/artifacts/evaluation'
    states = {'new_only': base.load(prior / 'new_only-e20.npz')}
    report = {}
    for arm in ARMS:
        model = torch.load(ROOT / 'models' / arm / 'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = model['E'].cuda(), model['R'].cuda()
        states[arm] = {split: base.ranks(E, R, q, truth, arm=arm, split=split) for split, q in queries.items()}
        np.savez(out / f'{arm}.npz', **states[arm])
        progress('evaluation', arm=arm, seconds=time.monotonic() - started)
    for arm, state in states.items():
        report[arm] = {split: base.metrics(state[split], q) for split, q in queries.items()}
        for split in ['old_train', 'old_valid']:
            report[arm][split]['drops'] = {}
            for col, kind in enumerate(['raw', 'filtered_train_truths']):
                mask = A[split][:, col] <= 10
                report[arm][split]['drops'][kind] = {'eligible': int(mask.sum()), 'drops': int((mask & (state[split][:, col] > 10)).sum())}
    dump('evaluation/metrics.json', report)
    return report, time.monotonic() - started


def summarize(risk, quality, costs):
    lines = ['# 058 risk forecasting and selective protection', '',
             'Seed17 development exploration. Fixed 16384 TRAIN-answer pool, 128 candidates, 10 ordinary warmup epochs + 10 protected epochs.',
             'Learners fit only early TRAIN trajectory/query groups. Late query/time holdout is risk prediction validation, not independent dataset confirmation.', '',
             '| Predictor | Late AP (new drops) | Late recall at 10% (new drops) | Positive events |',
             '|---|---:|---:|---:|']
    for name, item in risk['groups']['late_heldout'].items():
        m = item['new_drop_among_current_safe']
        lines.append(f"| {name} | {m.get('average_precision')} | {m.get('0.1', {}).get('recall')} | {m['positives']} |")
    lines += ['', '| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old TRAIN drops | Old-head/new-gold ratio | Train seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    ordinary = quality['new_only']['new_valid']['head_old_gold_new']['filtered_train_truths']['mrr']
    for arm, item in quality.items():
        m = item['new_valid']
        lines.append(f"| {arm} | {m['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['drops']['filtered_train_truths']['drops']} | {item['old_train']['drops']['filtered_train_truths']['drops']} | {m['head_old_gold_new']['filtered_train_truths']['mrr']/ordinary:.6f} | {costs['training'].get(arm, {}).get('seconds', 'reused')} |")
    lines += ['', 'No global safety guarantee. Same query pool, same protected samples/strength rule; feature/routing costs are included in training.',
              'Full-pool scans generate experiment labels; they are not features of the cheap predictors. Common trajectory costs and learner fitting are separately disclosed.',
              'Exposed VALID used only after all training. No TEST, independent CPU audit, SHA closure, archive, or time limit.']
    (ROOT / 'results.md').write_text('\n'.join(lines) + '\n')
    dump('costs.json', costs)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    progress('starting')
    initial = torch.load(SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt', map_location='cpu', weights_only=True)
    d = base.load(SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old, new = d['old'], d['new']
    tape = base.load(SRC / 'matched_protection52_20261013/artifacts/tape/learning.npz')
    A = base.load(SRC / 'replay_strength53_20261013/artifacts/evaluation/A.npz')
    data, truth = cohort(initial, old, new, A)
    costs = {'trajectory': make_trajectory(initial, old, new, tape, data)}
    models, risk = fit_predictors(data)
    costs['learner_fit_seconds'] = risk['fit_seconds']
    prefix = torch.load(ROOT / 'cache/prefix10.pt', map_location='cpu', weights_only=True)
    coefficient = calibrate(prefix, initial, new, tape, data)
    costs['training'] = {}
    for arm in ARMS:
        costs['training'][arm] = train_arm(arm, initial, prefix, new, tape, data, models, coefficient)
    quality, seconds = evaluate(old, new, truth, A)
    costs['evaluation_seconds'] = seconds
    costs['total_seconds'] = time.monotonic() - started
    summarize(risk, quality, costs)
    progress('complete', seconds=time.monotonic() - started)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        progress('failed', error=repr(exc))
        raise
