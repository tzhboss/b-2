import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/anchor_route97_20261014')
import experiment97 as previous
prior, action, route, core, base = previous.prior, previous.action, previous.route, previous.core, previous.base
ROOT = Path(__file__).resolve().parent
previous.ROOT = previous.previous.ROOT = prior.ROOT = action.ROOT = route.ROOT = route.prev.ROOT = core.ROOT = base.ROOT = ROOT
N = prior.N
ARMS = ['uniform_tree_norm', 'random_tree_norm']
route.ARMS = ARMS
SOURCE = Path('/data/lc/untzh/anchor_route97_20261014')


def train(arm, initial, prefix, new, tape, data, space, anchors, targets, selections):
    tick = time.monotonic()
    E, R, opt = core.newmodel(initial, prefix)
    records = []
    for j in range(220, 440):
        chosen = selections[j-220]
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
        projected = prior.project((E.detach()[N:]-reference).double(), space)
        norm = float(projected.norm())
        target = targets[j-220]['accepted_anchor_displacement_norm']
        assert norm > 0 or target == 0
        scale = target/norm if norm else 0.
        with torch.no_grad():
            E[N:].copy_(reference+(projected*scale).to(E.dtype))
        actual = (E.detach()[N:]-reference).double()
        actual_norm = float(actual.norm())
        assert abs(actual_norm-target) <= 1e-5*max(1., target)
        changes = (space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j, epoch=int(tape['epochs'][j]), target_norm=target, actual_norm=actual_norm,
                            scale=scale, max_cross_train_score_difference=float(changes.abs().max())))
        if j == 439 or tape['epochs'][j+1] != tape['epochs'][j]:
            core.progress('train', arm=arm, epoch=int(tape['epochs'][j]), scale=scale, norm=actual_norm)
    assert torch.equal(E[:N].cpu(), initial['E'][:N]) and torch.equal(R.cpu(), initial['R'])
    out = ROOT/'models'/arm
    out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(), R=R.cpu().double()), out/'epoch-20.pt')
    base.write(out/'steps.json', records)
    report = dict(seconds=time.monotonic()-tick, steps=220, frozen_equal=True,
                  final_anchor_displacement_norm=records[-1]['actual_norm'],
                  min_scale=min(x['scale'] for x in records), max_scale=max(x['scale'] for x in records),
                  amplification_steps=sum(x['scale']>1 for x in records),
                  max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records),
                  max_norm_error=max(abs(x['actual_norm']-x['target_norm']) for x in records))
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
    space, ss = previous.previous.previous.load_space()
    anchors = torch.load('/data/lc/untzh/anchor_space94_20261014/cache/ordinary-anchors.pt', map_location='cpu', weights_only=True)
    targets = json.loads((SOURCE/'models/anchor_tree/steps.json').read_text())
    random_steps = json.loads((SOURCE/'models/anchor_random_pool/steps.json').read_text())
    assert [x['step'] for x in targets] == [x['step'] for x in random_steps] == list(range(220, 440))
    schedules = dict(uniform_tree_norm=data['selections'], random_tree_norm=[x['selected'] for x in random_steps])
    costs = dict(preparation_seconds=time.monotonic()-tick, historical_assets_not_free=True,
                 target_source=str(SOURCE/'models/anchor_tree/steps.json'), training={})
    for arm in ARMS:
        costs['training'][arm] = train(arm, initial, prefix, new, tape, data, space, anchors, targets, schedules[arm])
        core.dump('costs-running.json', costs)
    metrics, es = route.evaluate(truths)
    historical = json.loads((SOURCE/'summary.json').read_text())['metrics']
    for name, metric in historical.items():
        if name not in metrics:
            metrics[name] = metric
    costs.update(evaluation_seconds=es, total_seconds=time.monotonic()-tick)
    core.dump('summary.json', dict(metrics=metrics, costs=costs))
    core.progress('complete', seconds=costs['total_seconds'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        core.progress('failed', error=repr(exc))
        raise
