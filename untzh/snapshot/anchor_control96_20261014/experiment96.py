import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/anchor_confirm95_20261014')
import experiment95 as confirm
previous, source = confirm.previous, confirm.source
prior, action, route, core, base = previous.prior, previous.action, previous.route, previous.core, previous.base
ROOT = Path(__file__).resolve().parent
N = prior.N


def configure(out, training_source):
    previous.ROOT = confirm.ROOT = source.ROOT = prior.ROOT = action.ROOT = route.ROOT = route.prev.ROOT = core.ROOT = base.ROOT = out
    route.SOURCE = training_source
    route.ARMS = ['anchor_common_norm']


def train(initial, prefix, new, tape, data, anchors, targets):
    tick = time.monotonic()
    E, R, opt = core.newmodel(initial, prefix)
    records = []
    for j in range(220, 440):
        chosen = data['selections'][j-220]
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
        norm = float(raw.norm())
        target = targets[j-220]['accepted_anchor_displacement_norm']
        assert norm > 0 or target == 0
        scale = target/norm if norm else 0.
        with torch.no_grad():
            E[N:].copy_(reference+(raw*scale).to(E.dtype))
        actual = float((E.detach()[N:]-reference).double().norm())
        assert abs(actual-target) <= 1e-5*max(1., target)
        assert torch.isfinite(E).all()
        records.append(dict(step=j, epoch=int(tape['epochs'][j]), raw_norm=norm, target_norm=target,
                            actual_norm=actual, scale=scale))
        if j == 439 or tape['epochs'][j+1] != tape['epochs'][j]:
            core.progress('train', arm='anchor_common_norm', epoch=int(tape['epochs'][j]), norm=actual, scale=scale)
    assert torch.equal(E[:N].cpu(), initial['E'][:N]) and torch.equal(R.cpu(), initial['R'])
    out = route.ROOT/'models/anchor_common_norm'
    out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(), R=R.cpu().double()), out/'epoch-20.pt')
    base.write(out/'steps.json', records)
    report = dict(seconds=time.monotonic()-tick, steps=220, frozen_equal=True,
                  final_anchor_displacement_norm=records[-1]['actual_norm'],
                  min_scale=min(x['scale'] for x in records), max_scale=max(x['scale'] for x in records),
                  amplification_steps=sum(x['scale']>1 for x in records),
                  max_norm_error=max(abs(x['actual_norm']-x['target_norm']) for x in records))
    base.write(out/'training.json', report)
    return report


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    specs = [(29, Path('/data/lc/untzh/risk_increment88_20261014'), Path('/data/lc/untzh/anchor_space94_20261014')),
             (17, Path('/data/lc/untzh/increment87_20261014'), Path('/data/lc/untzh/anchor_confirm95_20261014'))]
    bundles = []
    total_tick = time.monotonic()
    # Commit both fixed seed models before any new VALID evaluation.
    for seed, learning_source, reference_source in specs:
        out = ROOT/f'seed{seed}'
        out.mkdir()
        configure(out, learning_source)
        torch.manual_seed(seed)
        core.progress('starting')
        tick = time.monotonic()
        if seed == 29:
            initial, prefix, old, new, truths, tape, data, model, coefficient, ps = route.prepare()
            action.extra_data(truths, data)
        else:
            initial, prefix, new, tape, truths, data, ps = source.prepare()
        anchors = torch.load(reference_source/'cache/ordinary-anchors.pt', map_location='cpu', weights_only=True)
        assert anchors.shape == (220, 1876, 200)
        targets = json.loads((reference_source/'models/anchor_null/steps.json').read_text())
        assert [x['step'] for x in targets] == list(range(220, 440))
        costs = dict(preparation_seconds=time.monotonic()-tick, seed=seed, training_source=str(learning_source),
                     reference_source=str(reference_source), historical_anchors_not_free=True)
        costs['training'] = train(initial, prefix, new, tape, data, anchors, targets)
        core.dump('costs-running.json', costs)
        bundles.append((seed, learning_source, reference_source, out, truths, costs))
    summaries = {}
    for seed, learning_source, reference_source, out, truths, costs in bundles:
        configure(out, learning_source)
        metrics, es = route.evaluate(truths)
        historical = json.loads((reference_source/'summary.json').read_text())['metrics']
        for name, metric in historical.items():
            if name not in metrics:
                metrics[name] = metric
        costs['evaluation_seconds'] = es
        summary = dict(metrics=metrics, costs=costs)
        core.dump('summary.json', summary)
        core.progress('complete')
        summaries[str(seed)] = summary
    base.write(ROOT/'summary.json', dict(seeds=summaries, total_seconds=time.monotonic()-total_tick))
    base.write(ROOT/'status.json', dict(stage='complete', pid=os.getpid(), seconds=time.monotonic()-total_tick))
    print('COMPLETE96', time.monotonic()-total_tick, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        core.progress('failed', error=repr(exc))
        raise
