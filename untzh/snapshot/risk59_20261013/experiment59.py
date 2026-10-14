import collections
import copy
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn import functional as F

PRIOR = Path('/data/lc/untzh/risk58_20261013')
sys.path.insert(0, str(PRIOR))
import experiment58 as ex
ROOT = Path(__file__).resolve().parent
ex.ROOT = ROOT
ex.base.ROOT = ROOT
ARMS = {'uniform_top8': ('uniform', 8), 'current_all128': ('tree_current', 128),
        'current_top8': ('tree_current', 8), 'new_event_all128': ('tree_new_event', 128),
        'new_event_top8': ('tree_new_event', 8)}


def train(name, policy, k, initial, prefix, new, tape, data, predictors, coefficient):
    start = time.monotonic()
    E, R, opt = ex.init_model(initial)
    E.data.copy_(prefix['E'].cuda())
    opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    rng = np.random.default_rng(17)
    records, selection_seconds = [], 0.
    top = np.arange(ex.SIZE)
    for j in range(370, 740):
        epoch = int(tape['epochs'][j])
        if j == 370 or tape['epochs'][j - 1] != epoch:
            tick = time.monotonic()
            if policy != 'uniform':
                if policy == 'tree_new_event':
                    _, g0 = ex.gradient(E, R, new, tape, j)
                    delta = ex.proposal(E, opt, g0)
                else:
                    delta = torch.zeros_like(E)
                X, _ = ex.features(E, delta, data, epoch - 1)
                scores = predictors[policy].predict_proba(X)[:, 1]
                top = np.argsort(-scores, kind='stable')[:int(np.ceil(.2 * ex.SIZE))]
            ex.sync(); selection_seconds += time.monotonic() - tick
        selected = rng.choice(ex.SIZE, 32, replace=False) if policy == 'uniform' else np.concatenate([
            rng.choice(top, 26, replace=False), rng.choice(ex.SIZE, 6, replace=False)])
        base_loss, gl = ex.gradient(E, R, new, tape, j)
        ids = torch.as_tensor(data['candidates'][selected], device='cuda')
        q = torch.as_tensor(data['q'][selected], device='cuda')
        scores = (q[:, None] * E[ids].double()).sum(-1)
        if k < 128:
            scores = scores.topk(k, dim=-1).values
        penalty = F.softplus(scores).mean()
        gp = torch.autograd.grad(penalty, E)[0]
        bn, pn = float(gl[ex.N:].double().norm()), float(gp[ex.N:].double().norm())
        alpha = min(coefficient, .1 * bn / pn) if pn else coefficient
        combined = gl + alpha * gp
        if not torch.isfinite(base_loss) or not torch.isfinite(penalty) or not torch.isfinite(combined).all():
            raise FloatingPointError('nonfinite selective-candidate update')
        opt.zero_grad(set_to_none=True); E.grad = combined; opt.step()
        records.append({'step': j, 'epoch': epoch, 'learning_loss': float(base_loss),
                        'protection_loss': float(penalty), 'alpha': alpha,
                        'gradient_ratio': alpha * pn / bn if bn else 0.})
        if j == 739 or tape['epochs'][j + 1] != epoch:
            ex.progress('train', arm=name, epoch=epoch, seconds=time.monotonic() - start)
    assert torch.equal(E[:ex.N].detach().cpu(), initial['E'][:ex.N]) and torch.equal(R.cpu(), initial['R'])
    out = ROOT / 'models' / name; out.mkdir(parents=True, exist_ok=True)
    torch.save({'E': E.detach().cpu().double(), 'R': R.cpu().double()}, out / 'epoch-20.pt')
    ex.base.write(out / 'steps.json', records)
    result = {'seconds': time.monotonic() - start, 'selection_seconds_included': selection_seconds,
              'candidate_count': k, 'selector': policy, 'steps': 370,
              'max_gradient_ratio': max(r['gradient_ratio'] for r in records)}
    ex.base.write(out / 'training.json', result)
    return result


def evaluate(old, new, A):
    start = time.monotonic()
    truth = collections.defaultdict(set)
    for h, r, t in np.concatenate([old, new]):
        truth[int(h), int(r)].add(int(t))
    valid = ex.base.load(ex.SRC / 'entity_refresh43_20261012/artifacts/data/validation.npz')
    queries = {'old_train': old, 'old_valid': valid['old'], 'new_valid': valid['new']}
    out = ROOT / 'evaluation'; out.mkdir(exist_ok=True)
    states = {'new_only': ex.base.load(ex.SRC / 'replay_strength53_20261013/artifacts/evaluation/new_only-e20.npz'),
              'uniform_all128': ex.base.load(PRIOR / 'evaluation/uniform.npz')}
    for name in ARMS:
        model = torch.load(ROOT / 'models' / name / 'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = model['E'].cuda(), model['R'].cuda()
        state = {split: ex.base.ranks(E, R, q, truth, arm=name, split=split) for split, q in queries.items()}
        np.savez(out / f'{name}.npz', **state)
        states[name] = state
        ex.progress('evaluation', arm=name, seconds=time.monotonic() - start)
    report = {}
    for name, state in states.items():
        report[name] = {split: ex.base.metrics(state[split], q) for split, q in queries.items()}
        for split in ['old_train', 'old_valid']:
            report[name][split]['drops'] = {}
            for col, kind in enumerate(['raw', 'filtered_train_truths']):
                mask = A[split][:, col] <= 10
                report[name][split]['drops'][kind] = {'eligible': int(mask.sum()),
                    'drops': int((mask & (state[split][:, col] > 10)).sum())}
    ex.dump('evaluation/metrics.json', report)
    return report, time.monotonic() - start


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start = time.monotonic(); ex.progress('starting')
    initial = torch.load(ex.SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt', map_location='cpu', weights_only=True)
    d = ex.base.load(ex.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old, new = d['old'], d['new']
    tape = ex.base.load(ex.SRC / 'matched_protection52_20261013/artifacts/tape/learning.npz')
    A = ex.base.load(ex.SRC / 'replay_strength53_20261013/artifacts/evaluation/A.npz')
    data = ex.base.load(PRIOR / 'cache/cohort.npz')
    prefix = torch.load(PRIOR / 'cache/prefix10.pt', map_location='cpu', weights_only=True)
    models = {name: joblib.load(PRIOR / 'cache' / f'{name}.joblib') for name in ['tree_current', 'tree_new_event']}
    coefficient = json.loads((PRIOR / 'coefficient.json').read_text())['lambda']
    costs = {'reused_cache': str(PRIOR / 'cache'), 'new_preparation': 'load only; prior generation costs retained', 'training': {}}
    for name, (policy, k) in ARMS.items():
        costs['training'][name] = train(name, policy, k, initial, prefix, new, tape, data, models, coefficient)
    quality, seconds = evaluate(old, new, A)
    costs['evaluation_seconds'] = seconds; costs['total_seconds'] = time.monotonic() - start
    ex.dump('costs.json', costs)
    ordinary = quality['new_only']['new_valid']
    lines = ['# 059: candidate-directed simple protection', '',
             'Seed17. Same ten-epoch warmup/Adam state and cached learners as058. Full128 vs top8 sampled candidates; no new label preparation.', '',
             '| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old TRAIN drops | Old→new ratio | New→old ratio | New→new ratio |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, item in quality.items():
        nv = item['new_valid']
        ratios = [nv[d]['filtered_train_truths']['mrr'] / ordinary[d]['filtered_train_truths']['mrr'] for d in ['head_old_gold_new', 'head_new_gold_old', 'both_new']]
        lines.append(f"| {name} | {nv['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['drops']['filtered_train_truths']['drops']} | {item['old_train']['drops']['filtered_train_truths']['drops']} | {ratios[0]:.6f} | {ratios[1]:.6f} | {ratios[2]:.6f} |")
    lines += ['', 'Ordinary and uniform/all128 reused, not independent repetitions. Exposed VALID, no TEST. No hard safety or novelty claim.']
    (ROOT / 'results.md').write_text('\n'.join(lines) + '\n')
    ex.progress('complete', seconds=time.monotonic() - start)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        ex.progress('failed', error=repr(exc)); raise
