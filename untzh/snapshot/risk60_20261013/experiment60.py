import collections
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn import functional as F

PRIOR = Path('/data/lc/untzh/risk58_20261013')
PREV = Path('/data/lc/untzh/risk59_20261013')
sys.path.insert(0, str(PRIOR))
import experiment58 as ex
ROOT = Path(__file__).resolve().parent
ex.ROOT = ROOT; ex.base.ROOT = ROOT
ARMS = {'uniform_euclidean': ('uniform', 'euclidean'), 'risk_euclidean': ('risk', 'euclidean'),
        'uniform_adammetric': ('uniform', 'adammetric'), 'risk_adammetric': ('risk', 'adammetric')}


def direction_gradient(E, R, new, tape, j):
    lo, hi = tape['bounds'][j:j + 2]
    p = new[tape['pos'][lo:hi]]
    mask = (p[:, 0] < ex.N) & (p[:, 2] >= ex.N)
    if not mask.any():
        return torch.zeros_like(E)
    p, nh, nt = [torch.as_tensor(a, device='cuda', dtype=torch.long) for a in
                 [p[mask], tape['nh'][lo:hi][mask], tape['nt'][lo:hi][mask]]]
    h, r, t = p.unbind(1)
    pos = (E[h] * R[r] * E[t]).sum(-1)
    neg = (E[nh] * R[r[:, None]] * E[nt]).sum(-1)
    loss = F.softplus(-pos).mean() + F.softplus(neg).mean() + .0005 * (E[h].square().mean() + R[r].square().mean())
    return torch.autograd.grad(loss, E)[0]


def train(name, policy, metric, initial, prefix, new, tape, data, tree, coefficient):
    start = time.monotonic()
    E, R, opt = ex.init_model(initial)
    E.data.copy_(prefix['E'].cuda()); opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    rng = np.random.default_rng(17)
    records, projected, selection_seconds = [], 0, 0.
    top = np.arange(ex.SIZE)
    for j in range(370, 740):
        epoch = int(tape['epochs'][j])
        if j == 370 or tape['epochs'][j - 1] != epoch:
            tick = time.monotonic()
            if policy == 'risk':
                X, _ = ex.features(E, torch.zeros_like(E), data, epoch - 1)
                top = np.argsort(-tree.predict_proba(X)[:, 1], kind='stable')[:int(np.ceil(.2 * ex.SIZE))]
            ex.sync(); selection_seconds += time.monotonic() - tick
        selected = rng.choice(ex.SIZE, 32, replace=False) if policy == 'uniform' else np.concatenate([
            rng.choice(top, 26, replace=False), rng.choice(ex.SIZE, 6, replace=False)])
        loss, gl = ex.gradient(E, R, new, tape, j)
        penalty = ex.protection(E, data, selected)
        gp = torch.autograd.grad(penalty, E)[0]
        gd = direction_gradient(E, R, new, tape, j)
        a, b = gp[ex.N:].double(), gd[ex.N:].double()
        if metric == 'adammetric':
            state = opt.state[E]
            step = float(state['step']) + 1
            variance = .999 * state['exp_avg_sq'][ex.N:].double() + .001 * gl[ex.N:].double().square()
            diagonal = (variance.sqrt() / np.sqrt(1 - .999 ** step) + 1e-8).reciprocal()
        else:
            diagonal = torch.ones_like(a)
        inner = (a * diagonal * b).sum()
        norm2 = (b.square() * diagonal).sum()
        changed = bool(inner < 0 and norm2 > 0)
        if changed:
            a = a - (inner / norm2) * b
            gp = torch.cat([torch.zeros_like(gp[:ex.N]), a.to(gp.dtype)])
            projected += 1
        after = float((a * diagonal * b).sum())
        bn, pn = float(gl[ex.N:].double().norm()), float(gp[ex.N:].double().norm())
        alpha = min(coefficient, .1 * bn / pn) if pn else coefficient
        combined = gl + alpha * gp
        if not torch.isfinite(loss) or not torch.isfinite(penalty) or not torch.isfinite(combined).all():
            raise FloatingPointError('nonfinite coordinated update')
        opt.zero_grad(set_to_none=True); E.grad = combined; opt.step()
        records.append({'step': j, 'epoch': epoch, 'loss': float(loss), 'penalty': float(penalty),
                        'projected': changed, 'inner_before': float(inner), 'inner_after': after,
                        'alpha': alpha, 'gradient_ratio': alpha * pn / bn if bn else 0.})
        if j == 739 or tape['epochs'][j + 1] != epoch:
            ex.progress('train', arm=name, epoch=epoch, projected_steps=projected, seconds=time.monotonic() - start)
    assert torch.equal(E[:ex.N].detach().cpu(), initial['E'][:ex.N]) and torch.equal(R.cpu(), initial['R'])
    out = ROOT / 'models' / name; out.mkdir(parents=True, exist_ok=True)
    torch.save({'E': E.detach().cpu().double(), 'R': R.cpu().double()}, out / 'epoch-20.pt')
    ex.base.write(out / 'steps.json', records)
    result = {'seconds': time.monotonic() - start, 'selection_seconds_included': selection_seconds,
              'steps': 370, 'projected_steps': projected, 'metric': metric,
              'max_gradient_ratio': max(r['gradient_ratio'] for r in records)}
    ex.base.write(out / 'training.json', result)
    return result


def evaluate(old, new, A):
    started = time.monotonic()
    truth = collections.defaultdict(set)
    for h, r, t in np.concatenate([old, new]):
        truth[int(h), int(r)].add(int(t))
    valid = ex.base.load(ex.SRC / 'entity_refresh43_20261012/artifacts/data/validation.npz')
    prior_metrics = json.loads((PRIOR / 'evaluation/metrics.json').read_text())
    previous_metrics = json.loads((PREV / 'evaluation/metrics.json').read_text())
    report = {'new_only': prior_metrics['new_only'], 'uniform': prior_metrics['uniform'],
              'risk_current': previous_metrics['current_all128']}
    out = ROOT / 'evaluation'; out.mkdir(exist_ok=True)
    chosen = []
    valid_seconds = 0.
    for name in ARMS:
        tick = time.monotonic()
        model = torch.load(ROOT / 'models' / name / 'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = model['E'].cuda(), model['R'].cuda()
        ranks = {split: ex.base.ranks(E, R, valid[key], truth, arm=name, split=split)
                 for split, key in [('old_valid', 'old'), ('new_valid', 'new')]}
        np.savez(out / f'{name}-valid.npz', **ranks)
        report[name] = {split: ex.base.metrics(ranks[split], valid[key])
                        for split, key in [('old_valid', 'old'), ('new_valid', 'new')]}
        report[name]['old_valid']['drops'] = {}
        for col, kind in enumerate(['raw', 'filtered_train_truths']):
            mask = A['old_valid'][:, col] <= 10
            report[name]['old_valid']['drops'][kind] = {'eligible': int(mask.sum()),
                'drops': int((mask & (ranks['old_valid'][:, col] > 10)).sum())}
        nv, baseline = report[name]['new_valid'], report['new_only']['new_valid']
        ratios = [nv[d]['filtered_train_truths']['mrr'] / baseline[d]['filtered_train_truths']['mrr']
                  for d in ['head_old_gold_new', 'head_new_gold_old', 'both_new']]
        passed = min(ratios) >= .98 and report[name]['old_valid']['drops']['filtered_train_truths']['drops'] <= 5
        report[name]['quality_screen_pass'] = passed
        if passed:
            chosen.append(name)
        valid_seconds += time.monotonic() - tick
        ex.progress('valid_screen', arm=name, passed=passed, directional_ratios=ratios)
    for name in chosen:
        model = torch.load(ROOT / 'models' / name / 'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = model['E'].cuda(), model['R'].cuda()
        ranks = ex.base.ranks(E, R, old, truth, arm=name, split='old_train')
        np.savez(out / f'{name}-train.npz', old_train=ranks)
        report[name]['old_train'] = ex.base.metrics(ranks, old)
        report[name]['old_train']['drops'] = {}
        for col, kind in enumerate(['raw', 'filtered_train_truths']):
            mask = A['old_train'][:, col] <= 10
            report[name]['old_train']['drops'][kind] = {'eligible': int(mask.sum()),
                'drops': int((mask & (ranks[:, col] > 10)).sum())}
        ex.progress('full_train_evaluation', arm=name)
    ex.dump('evaluation/metrics.json', report)
    return report, {'total_seconds': time.monotonic() - started, 'valid_seconds': valid_seconds,
                     'full_train_arms': chosen}


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic(); ex.progress('starting')
    initial = torch.load(ex.SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt', map_location='cpu', weights_only=True)
    d = ex.base.load(ex.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old, new = d['old'], d['new']
    tape = ex.base.load(ex.SRC / 'matched_protection52_20261013/artifacts/tape/learning.npz')
    A = ex.base.load(ex.SRC / 'replay_strength53_20261013/artifacts/evaluation/A.npz')
    data = ex.base.load(PRIOR / 'cache/cohort.npz')
    prefix = torch.load(PRIOR / 'cache/prefix10.pt', map_location='cpu', weights_only=True)
    tree = joblib.load(PRIOR / 'cache/tree_current.joblib')
    coefficient = json.loads((PRIOR / 'coefficient.json').read_text())['lambda']
    costs = {'reused_cache': str(PRIOR / 'cache'), 'training': {}}
    for name, (policy, metric) in ARMS.items():
        costs['training'][name] = train(name, policy, metric, initial, prefix, new, tape, data, tree, coefficient)
    quality, evaluation = evaluate(old, new, A)
    costs['evaluation'] = evaluation; costs['total_seconds'] = time.monotonic() - started
    ex.dump('costs.json', costs)
    baseline = quality['new_only']['new_valid']
    lines = ['# 060: direction coordination after risk identification', '',
             'Single seed17 development experiment. Same warmup, query/candidate cache, loss, strength cap and classifier. One-direction projection is not a ranking safety guarantee.', '',
             '| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old→new ratio | New→old ratio | New→new ratio | Quality screen |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, item in quality.items():
        nv = item['new_valid']
        ratios = [nv[d]['filtered_train_truths']['mrr'] / baseline[d]['filtered_train_truths']['mrr'] for d in ['head_old_gold_new', 'head_new_gold_old', 'both_new']]
        lines.append(f"| {name} | {nv['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['drops']['filtered_train_truths']['drops']} | {ratios[0]:.6f} | {ratios[1]:.6f} | {ratios[2]:.6f} | {item.get('quality_screen_pass', 'reused')} |")
    lines += ['', 'All candidates receive VALID screening after training. Only quality-screen candidates receive expensive full TRAIN ranking. Failed candidates have TRAIN metrics uncomputed, not zero.',
              'Historical controls reused; no independent repetitions, TEST, audit/closure/archive, or fixed timeout.']
    (ROOT / 'results.md').write_text('\n'.join(lines) + '\n')
    ex.progress('complete', seconds=time.monotonic() - started)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        ex.progress('failed', error=repr(exc)); raise
