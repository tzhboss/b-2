import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/anchor_space94_20261014')
import experiment94 as previous
prior, action, route, core, base = previous.prior, previous.action, previous.route, previous.core, previous.base
ROOT = Path(__file__).resolve().parent
previous.ROOT = prior.ROOT = action.ROOT = route.ROOT = route.prev.ROOT = core.ROOT = base.ROOT = ROOT
route.SOURCE = Path('/data/lc/untzh/risk_increment88_20261014')
N, SIZE = prior.N, core.SIZE
ARMS = ['anchor_tree', 'anchor_pressure128', 'anchor_random_pool']
route.ARMS = ARMS


def train(arm, initial, prefix, new, tape, data, model, space, anchors):
    tick = time.monotonic()
    E, R, opt = core.newmodel(initial, prefix)
    rng = np.random.default_rng(890029)
    records, decisions = [], []
    detection_seconds = 0.
    projection_seconds = 0.
    for j in range(220, 440):
        ep = int(tape['epochs'][j])-1
        if j == 220 or tape['epochs'][j-1] != tape['epochs'][j]:
            torch.cuda.synchronize()
            start = time.monotonic()
            if arm == 'anchor_random_pool':
                scores = np.random.default_rng(890000+ep).random(SIZE)
            else:
                X = route.features(E, data, ep)
                scores = model.predict_proba(X)[:,1] if arm == 'anchor_tree' else X[:,9]-np.log(X[:,1])
                if j == 220:
                    frame = base.load(route.SOURCE/'cache/frame-10.npz')['X']
                    cols = list(range(15))+[20]
                    assert np.max(np.abs(X[:,cols]-frame[:,cols])) <= 1e-6
            pool = np.argsort(-scores, kind='stable')[:int(np.ceil(.05*SIZE))]
            torch.cuda.synchronize()
            seconds = time.monotonic()-start
            detection_seconds += seconds
            decisions.append(dict(epoch=ep, seconds=seconds, pool=pool.tolist()))
        first = rng.choice(pool, 26, replace=False)
        available = np.ones(SIZE, bool)
        available[first] = False
        chosen = np.concatenate([first, rng.choice(np.flatnonzero(available), 6, replace=False)])
        loss = base.learning(E, R, new, tape, j)
        gl = torch.autograd.grad(loss, E)[0]
        p, _ = action.penalty(E, data, chosen, 'absolute_full')
        gp = torch.autograd.grad(p, E)[0]
        bn, pn = float(gl[N:].double().norm()), float(gp[N:].double().norm())
        alpha = .1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True)
        E.grad = gl+alpha*gp
        opt.step()
        reference = anchors[j-220].cuda()
        raw = (E.detach()[N:]-reference).double()
        torch.cuda.synchronize()
        start = time.monotonic()
        projected = prior.project(raw, space)
        projection_seconds += time.monotonic()-start
        assert float(projected.norm()) <= float(raw.norm())*(1+1e-8)+1e-12
        with torch.no_grad():
            E[N:].copy_(reference+projected.to(E.dtype))
        actual = (E.detach()[N:]-reference).double()
        changes = (space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j, epoch=ep+1, selected=chosen.tolist(),
                            raw_anchor_displacement_norm=float(raw.norm()),
                            accepted_anchor_displacement_norm=float(actual.norm()),
                            max_cross_train_score_difference=float(changes.abs().max()),
                            gradient_ratio=alpha*pn/bn if bn else 0.))
        if j == 439 or tape['epochs'][j+1] != tape['epochs'][j]:
            core.progress('train', arm=arm, epoch=ep+1, max_score_difference=records[-1]['max_cross_train_score_difference'])
    assert torch.equal(E[:N].cpu(), initial['E'][:N]) and torch.equal(R.cpu(), initial['R'])
    out = ROOT/'models'/arm
    out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(), R=R.cpu().double()), out/'epoch-20.pt')
    base.write(out/'steps.json', records)
    base.write(out/'decisions.json', decisions)
    report = dict(seconds=time.monotonic()-tick, detection_seconds_included=detection_seconds,
                  projection_seconds_included=projection_seconds, steps=220, frozen_equal=True,
                  max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records),
                  final_anchor_displacement_norm=records[-1]['accepted_anchor_displacement_norm'],
                  mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in records])))
    base.write(out/'training.json', report)
    return report


def main():
    torch.set_num_threads(2)
    torch.manual_seed(29)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tick = time.monotonic()
    core.progress('starting')
    initial, prefix, old, new, truths, tape, data, model, coefficient, ps = route.prepare()
    es = action.extra_data(truths, data)
    space, ss = previous.previous.load_space()
    start = time.monotonic()
    anchors = torch.load('/data/lc/untzh/anchor_space94_20261014/cache/ordinary-anchors.pt', map_location='cpu', weights_only=True)
    assert anchors.shape == (220, 1876, 200)
    costs = dict(preparation_seconds=ps+es+ss+time.monotonic()-start,
                 reused_anchors='/data/lc/untzh/anchor_space94_20261014/cache/ordinary-anchors.pt',
                 reused_space='/data/lc/untzh/query_space92_20261014/cache/query-space.pt',
                 historical_assets_not_free=True, training={})
    for arm in ARMS:
        costs['training'][arm] = train(arm, initial, prefix, new, tape, data, model, space, anchors)
        core.dump('costs-running.json', costs)
    metrics, es = route.evaluate(truths)
    historical = json.loads(Path('/data/lc/untzh/anchor_space94_20261014/summary.json').read_text())['metrics']
    for arm, metric in historical.items():
        if arm not in metrics:
            metrics[arm] = metric
    costs.update(evaluation_seconds=es, total_seconds=time.monotonic()-tick)
    core.dump('summary.json', dict(metrics=metrics, costs=costs))
    core.progress('complete', seconds=costs['total_seconds'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        core.progress('failed', error=repr(exc))
        raise
