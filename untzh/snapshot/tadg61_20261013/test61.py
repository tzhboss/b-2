"""TADG formula applicability and topology features on cached real experiment."""
import collections
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from torch.nn import functional as F

PRIOR = Path('/data/lc/untzh/risk58_20261013')
sys.path.insert(0, str(PRIOR))
import experiment58 as ex
ROOT = Path(__file__).resolve().parent
ex.ROOT = ROOT; ex.base.ROOT = ROOT


def topology(old):
    start = time.monotonic()
    # Each even relation ID represents an original TRAIN fact; remove artificial inverse records.
    canonical = old[old[:, 1] % 2 == 0]
    adjacency = [collections.defaultdict(float) for _ in range(ex.N)]
    for h, r, t in canonical:
        adjacency[int(h)][int(t)] += .15
        adjacency[int(t)][int(h)] += .85
    entropy = np.zeros(ex.N)
    degrees = np.zeros(ex.N, dtype=int)
    for i, links in enumerate(adjacency):
        degrees[i] = len(links)
        if links:
            p = np.asarray(list(links.values()), dtype=float)
            p /= p.sum()
            entropy[i] = -(p * np.log2(p)).sum()
    weights = 1 / (1 + np.exp(-entropy))
    np.savez(ROOT / 'topology.npz', entropy=entropy, weights=weights, degrees=degrees)
    ex.dump('topology.json', {'seconds': time.monotonic() - start, 'lambda': .85,
        'source': 'previous-snapshot TRAIN only, original directed facts, counts across relations',
        'canonical_train_facts': len(canonical), 'old_entities': ex.N,
        'entropy_quantiles': np.quantile(entropy, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        'weight_quantiles': np.quantile(weights, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        'weight_ge_098': int((weights >= .98).sum()), 'weight_ge_099': int((weights >= .99).sum()),
        'degree_entropy_correlation': float(np.corrcoef(np.log1p(degrees), entropy)[0, 1])})
    return entropy, weights


def score(E, R, triples):
    h, r, t = triples.unbind(1)
    return (E[h] * R[r] * E[t]).sum(-1)


def formula_check(initial, old, weights):
    rng = np.random.default_rng(17)
    mem = torch.as_tensor(old[rng.choice(len(old), 512, replace=False)], device='cuda')
    teacher_E, teacher_R = initial['E'].cuda().double(), initial['R'].cuda().double()
    teacher = score(teacher_E, teacher_R, mem).detach()
    w = torch.as_tensor(weights, device='cuda')
    paths = {'initial': None, 'ordinary_prefix10': PRIOR / 'cache/prefix10.pt',
             'ordinary_epoch20': ex.SRC / 'matched_protection52_20261013/artifacts/train/new_only/epoch-20.pt',
             'risk_current_epoch20': Path('/data/lc/untzh/risk59_20261013/models/current_all128/epoch-20.pt')}
    result = {}
    for name, path in paths.items():
        state = initial if path is None else torch.load(path, map_location='cpu', weights_only=True)
        E = torch.nn.Parameter(state['E'].cuda().double().clone())
        R = torch.nn.Parameter(state['R'].cuda().double().clone())
        distill = F.huber_loss(score(E, R, mem), teacher, delta=1., reduction='mean')
        ge, gr = torch.autograd.grad(distill, [E, R])
        tar = (w[:, None] * (E[:ex.N] - teacher_E[:ex.N]).square()).sum()
        gt = torch.autograd.grad(tar, E)[0]
        result[name] = {'distillation_loss': float(distill), 'reference_E_grad_norm': float(ge.norm()),
            'reference_R_grad_norm': float(gr.norm()), 'reference_new_rows_grad_norm': float(ge[ex.N:].norm()),
            'TAR_loss': float(tar), 'TAR_grad_norm': float(gt.norm()),
            'old_E_equal': bool(torch.equal(E[:ex.N], teacher_E[:ex.N])), 'old_R_equal': bool(torch.equal(R, teacher_R))}
        assert float(distill) == 0 and float(tar) == 0 and torch.count_nonzero(ge) == 0 and torch.count_nonzero(gr) == 0 and torch.count_nonzero(gt) == 0
    # Deliberately perturb a historical row as a responsiveness check, not a trained method.
    E = torch.nn.Parameter(teacher_E.clone())
    with torch.no_grad():
        E[int(mem[0, 0]), 0] += .01
    loss = F.huber_loss(score(E, teacher_R, mem), teacher, delta=1.)
    grad = torch.autograd.grad(loss, E)[0]
    result['old_row_perturbation_control'] = {'distillation_loss': float(loss),
        'old_grad_norm': float(grad[:ex.N].norm()), 'new_grad_norm': float(grad[ex.N:].norm())}
    assert loss > 0 and grad[:ex.N].norm() > 0 and torch.count_nonzero(grad[ex.N:]) == 0
    ex.dump('formula-applicability.json', result)


def risk_features(entropy, weights):
    start = time.monotonic()
    data = ex.base.load(PRIOR / 'cache/cohort.npz')
    h, t = data['queries'][:, 0], data['queries'][:, 2]
    extra = np.column_stack([entropy[h], entropy[t], weights[h], weights[t]]).astype(np.float32)
    frames = [ex.base.load(PRIOR / 'cache' / f'frame-{i:02d}.npz') for i in range(21)]
    train = data['split'] == 0
    X = np.concatenate([np.column_stack([frames[i]['X'], extra])[train] for i in range(8)])
    y = np.concatenate([(frames[i + 1]['ranks'][train] > 10).astype(int) for i in range(8)])
    sample_weights = np.where(y, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))
    models = {'current_tree': joblib.load(PRIOR / 'cache/tree_current.joblib')}
    definitions = {'current_plus_raw_entropy': list(range(15)) + [20, 21, 22],
                   'current_plus_sigmoid_entropy': list(range(15)) + [20, 23, 24],
                   'topology_only_tree': [21, 22, 23, 24]}
    fit_seconds = {}
    for name, cols in definitions.items():
        model = make_pipeline(ColumnTransformer([('features', 'passthrough', cols)]),
            HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, max_depth=3,
                learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=17))
        tick = time.monotonic()
        model.fit(X, y, histgradientboostingclassifier__sample_weight=sample_weights)
        fit_seconds[name] = time.monotonic() - tick
        joblib.dump(model, ROOT / f'{name}.joblib')
        models[name] = model
    outcomes = collections.defaultdict(list)
    test = data['split'] == 2
    for epoch in range(10, 20):
        features = np.column_stack([frames[epoch]['X'], extra])[test]
        target = frames[epoch + 1]['ranks'][test] > 10
        safe = frames[epoch]['ranks'][test] <= 10
        new = target & safe
        scores = {'entropy_importance': extra[test, :2].max(1),
                  'sigmoid_entropy_importance': extra[test, 2:].max(1)}
        for name, model in models.items():
            Xi = frames[epoch]['X'][test] if name == 'current_tree' else features
            scores[name] = model.predict_proba(Xi)[:, 1]
        for name, value in scores.items():
            ap = ex.selection_metrics(target, value, safe)['average_precision']
            chosen = np.argsort(-value, kind='stable')[:int(np.ceil(.1 * len(value)))]
            outcomes[name].append({'epoch': epoch, 'new_drop_events': int(new.sum()),
                'selected': len(chosen), 'new_hits': int(new[chosen].sum()), 'safe_AP': ap})
    summary = {}
    for name, rows in outcomes.items():
        total, hits = sum(r['new_drop_events'] for r in rows), sum(r['new_hits'] for r in rows)
        summary[name] = {'new_hits': hits, 'new_events': total, 'recall10': hits / total,
            'macro_safe_AP': float(np.mean([r['safe_AP'] for r in rows if r['safe_AP'] is not None])),
            'per_epoch': rows}
    ex.dump('entropy-risk.json', {'fit_seconds': fit_seconds, 'methods': summary,
        'seconds': time.monotonic() - start, 'protocol': 'same early TRAIN fit and late query/time holdout as058; no VALID/TEST'})


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start = time.monotonic(); ex.progress('starting')
    initial = torch.load(ex.SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt', map_location='cpu', weights_only=True)
    old = ex.base.load(ex.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')['old']
    entropy, weights = topology(old)
    formula_check(initial, old, weights)
    ex.progress('formula_checked')
    risk_features(entropy, weights)
    ex.progress('complete', seconds=time.monotonic() - start)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        ex.progress('failed', error=repr(exc)); raise
