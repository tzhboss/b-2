import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import json
from pathlib import Path
import torch
sys.path.insert(0, '/data/lc/untzh/anchor_space94_20261014')
import experiment94 as previous
sys.path.insert(0, '/data/lc/untzh/rank_confirm91_20261014')
import experiment91 as source
prior, action, route, core, base = previous.prior, previous.action, previous.route, previous.core, previous.base
ROOT = Path(__file__).resolve().parent
previous.ROOT = source.ROOT = prior.ROOT = action.ROOT = route.ROOT = route.prev.ROOT = core.ROOT = base.ROOT = ROOT
route.SOURCE = source.SOURCE
N = prior.N
ARMS = ['anchor_null', 'anchor_normmatched']
route.ARMS = ARMS


def anchors(initial, prefix, new, tape):
    tick = time.monotonic()
    E, R, opt = core.newmodel(initial, prefix)
    states = []
    for j in range(220, 440):
        loss = base.learning(E, R, new, tape, j)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        states.append(E.detach()[N:].cpu().clone())
        if j == 439 or tape['epochs'][j+1] != tape['epochs'][j]:
            core.progress('prepare_anchor', epoch=int(tape['epochs'][j]))
    ordinary = torch.load(route.SOURCE/'models/none/epoch-20.pt', map_location='cpu', weights_only=True)
    assert torch.equal(E.detach().cpu().double(), ordinary['E'])
    assert torch.equal(R.cpu().double(), ordinary['R'])
    states = torch.stack(states)
    (ROOT/'cache').mkdir(exist_ok=True)
    torch.save(states, ROOT/'cache/ordinary-anchors.pt')
    return states, dict(seconds=time.monotonic()-tick, bytes=states.numel()*states.element_size(), ordinary_endpoint_equal=True)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tick = time.monotonic()
    core.progress('starting')
    initial, prefix, new, tape, truths, data, ps = source.prepare()
    space, ss = prior.build_space(initial, new)
    states, ac = anchors(initial, prefix, new, tape)
    costs = dict(preparation_seconds=ps, space=ss, anchors=ac, seed=17,
                 source=str(route.SOURCE), formula='fixed094 cumulative-null and own-state normmatched; fullabsolute rho .1; joint Adam', training={})
    for arm in ARMS:
        costs['training'][arm] = previous.train(arm, initial, prefix, new, tape, data, space, states)
        core.dump('costs-running.json', costs)
    metrics, es = route.evaluate(truths)
    historical = json.loads(Path('/data/lc/untzh/rank_confirm91_20261014/summary.json').read_text())['metrics']
    for arm in ['absolute128_uniform', 'absolute_full_uniform', 'violation_full_uniform', 'budget_full_uniform']:
        metrics[arm] = historical[arm]
    costs.update(evaluation_seconds=es, total_seconds=time.monotonic()-tick)
    core.dump('summary.json', dict(metrics=metrics, costs=costs))
    core.progress('complete', seconds=costs['total_seconds'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        core.progress('failed', error=repr(exc))
        raise
