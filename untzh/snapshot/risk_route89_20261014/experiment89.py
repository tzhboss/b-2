import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, '/data/lc/untzh/increment87_20261014')
import experiment87 as prev
core, base, ex = prev.core, prev.base, prev.ex
ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lc/untzh/risk_increment88_20261014')
prev.ROOT = core.ROOT = base.ROOT = ROOT
N, TOTAL, SIZE = prev.N, prev.TOTAL, core.SIZE
ARMS = ['transfer_tree', 'pressure128', 'pressure32', 'random_pool']


def prepare():
    tick = time.monotonic()
    initial = torch.load(SOURCE/'cache/initial.pt', map_location='cpu', weights_only=True)
    prefix = torch.load(SOURCE/'cache/prefix10.pt', map_location='cpu', weights_only=True)
    z = base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    old, new = z['old'], z['new']
    truths = core.truthmap(old, np.concatenate([new, z['extra']]))
    tape = base.load(SOURCE/'cache/learning.npz')
    assert len(tape['epochs']) == 440 and np.all(tape['epochs'][220:] >= 11)
    c = base.load(SOURCE/'cache/cohort.npz')
    r = base.load(SOURCE/'cache/risk-data.npz')
    data = dict(q=torch.as_tensor(c['q'], device='cuda'), candidates=torch.as_tensor(c['candidates'], device='cuda'),
                fixed=r['fixed'], gold=torch.as_tensor(r['gold'], device='cuda'), tau=torch.as_tensor(r['tau'], device='cuda'))
    model = joblib.load('/data/lc/untzh/risk58_20261013/cache/tree_current.joblib')
    coefficient = json.loads((SOURCE/'calibration.json').read_text())['coefficients']['absolute']
    assert len(c['queries']) == SIZE and np.all((c['candidates'] >= N) & (c['candidates'] < TOTAL))
    assert torch.equal(prefix['E'][:N], initial['E'][:N])
    for i, (h, rel, gold) in enumerate(c['queries']):
        assert not truths[int(h), int(rel)].intersection(c['candidates'][i].tolist())
    return initial, prefix, old, new, truths, tape, data, model, coefficient, time.monotonic()-tick


@torch.no_grad()
def features(E, data, ep, count=128):
    parts = []
    for lo in range(0, SIZE, 256):
        hi = min(lo+256, SIZE)
        q, tau, gold = data['q'][lo:hi], data['tau'][lo:hi], data['gold'][lo:hi]
        ids = data['candidates'][lo:hi, :count]
        z = ((q[:, None]*E[ids].double()).sum(-1)-gold[:, None])/tau[:, None]
        if count == 32:
            parts.append(z.max(-1).values.float().cpu().numpy())
        else:
            v = torch.stack([z[:, :32].max(-1).values, z.max(-1).values, z.topk(4, dim=-1).values.mean(-1),
                             z.mean(-1), z.std(-1), (z>0).double().mean(-1), F.softplus(z).mean(-1)], 1)
            parts.append(v.float().cpu().numpy())
    if count == 32:
        return np.concatenate(parts)
    X = np.zeros((SIZE, 21), np.float32)
    X[:, :8] = data['fixed']; X[:, 8:15] = np.concatenate(parts); X[:, 20] = ep/20
    assert np.isfinite(X).all()
    return X


def train(arm, initial, prefix, new, tape, data, model, coefficient):
    tick = time.monotonic(); E, R, opt = core.newmodel(initial, prefix)
    rng = np.random.default_rng(890029)
    steps, decisions, detection_seconds = [], [], 0.
    source_check = None
    for j in range(220, 440):
        ep = int(tape['epochs'][j])-1
        if j == 220 or tape['epochs'][j-1] != tape['epochs'][j]:
            torch.cuda.synchronize(); t = time.monotonic()
            if arm == 'random_pool':
                scores = np.random.default_rng(890000+ep).random(SIZE)
            elif arm == 'pressure32':
                scores = features(E, data, ep, 32)-np.log(data['fixed'][:, 1])
            else:
                X = features(E, data, ep)
                scores = model.predict_proba(X)[:, 1] if arm == 'transfer_tree' else X[:, 9]-np.log(X[:, 1])
                if j == 220:
                    source = base.load(SOURCE/'cache/frame-10.npz')['X']
                    cols = list(range(15))+[20]
                    source_check = float(np.max(np.abs(X[:, cols]-source[:, cols])))
                    assert source_check <= 1e-6
            pool = np.argsort(-scores, kind='stable')[:int(np.ceil(.05*SIZE))]
            torch.cuda.synchronize(); elapsed = time.monotonic()-t; detection_seconds += elapsed
            decisions.append(dict(epoch=ep, detection_seconds=elapsed, pool=pool.tolist()))
        chosen_risk = rng.choice(pool, 26, replace=False)
        available = np.ones(SIZE, bool); available[chosen_risk] = False
        chosen = np.concatenate([chosen_risk, rng.choice(np.flatnonzero(available), 6, replace=False)])
        loss = base.learning(E, R, new, tape, j)
        gl = torch.autograd.grad(loss, E)[0]
        gp = torch.autograd.grad(core.penalty(E, data, chosen, 'absolute'), E)[0]
        bn, pn = float(gl[N:].double().norm()), float(gp[N:].double().norm())
        alpha = min(coefficient, .1*bn/pn) if pn else 0.
        ratio = alpha*pn/bn if bn else 0.
        assert ratio <= .100001 and torch.isfinite(loss) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True); E.grad = gl+alpha*gp; opt.step()
        assert torch.isfinite(E).all()
        steps.append(dict(step=j, epoch=ep+1, alpha=alpha, gradient_ratio=ratio, selected=chosen.tolist()))
        if j == 439 or tape['epochs'][j+1] != tape['epochs'][j]:
            core.progress('train', arm=arm, epoch=ep+1)
    assert torch.equal(E[:N].cpu(), initial['E'][:N]) and torch.equal(R.cpu(), initial['R'])
    out = ROOT/'models'/arm; out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(), R=R.cpu().double()), out/'epoch-20.pt')
    base.write(out/'steps.json', steps); base.write(out/'decisions.json', decisions)
    report = dict(seconds=time.monotonic()-tick, detection_seconds_included=detection_seconds, steps=220,
                  source_feature_max_abs=source_check, mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in steps])),
                  max_gradient_ratio=max(x['gradient_ratio'] for x in steps), frozen_equal=True)
    base.write(out/'training.json', report)
    return report


def evaluate(truths):
    tick = time.monotonic(); out = ROOT/'evaluation'; out.mkdir()
    qs = base.load(SOURCE/'evaluation/queries.npz'); teacher = base.load(SOURCE/'evaluation/teacher.npz')
    historical = json.loads((SOURCE/'summary.json').read_text())['metrics']
    report = {arm:copy.deepcopy(historical[arm]) for arm in ['none', 'absolute', 'absolute075']}
    for arm in ARMS:
        m = torch.load(ROOT/'models'/arm/'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = m['E'].cuda(), m['R'].cuda()
        states = {k:base.ranks(E, R, q, truths, arm=arm, split=k) for k, q in qs.items()}
        np.savez(out/f'{arm}.npz', **states)
        report[arm] = {k:base.metrics(states[k], q) for k, q in qs.items()}
        for k in ['old_train', 'old_valid']:
            report[arm][k]['drops'] = core.drops(states[k], teacher[k])
        report[arm]['directional_ratios'] = [report[arm]['new_valid'][d]['filtered_train_truths']['mrr']/report['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        core.progress('evaluated', arm=arm, new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],
                      train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    core.dump('evaluation/metrics.json', report)
    return report, time.monotonic()-tick


def main():
    torch.set_num_threads(2); torch.manual_seed(29); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic(); core.progress('starting')
    initial, prefix, old, new, truths, tape, data, model, coefficient, ps = prepare()
    costs=dict(preparation_seconds=ps, reused_source=str(SOURCE), coefficient=coefficient, training={})
    for arm in ARMS:
        costs['training'][arm] = train(arm, initial, prefix, new, tape, data, model, coefficient)
        core.dump('costs-running.json', costs)
    report, es=evaluate(truths); costs.update(evaluation_seconds=es, total_seconds=time.monotonic()-tick)
    core.dump('summary.json', dict(metrics=report, costs=costs)); core.progress('complete', seconds=costs['total_seconds'])


if __name__ == '__main__':
    try: main()
    except Exception as exc: core.progress('failed', error=repr(exc)); raise
