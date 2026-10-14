"""TRAIN-only permission/stability mechanism screen; not native TADG reproduction."""
import collections
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PRIOR = Path('/data/lc/untzh/risk58_20261013')
sys.path.insert(0, str(PRIOR))
import experiment58 as ex
ROOT = Path(__file__).resolve().parent
ex.ROOT = ROOT; ex.base.ROOT = ROOT
ARMS = {
    'slow_frozen': (.0007, False, 'none'),
    'slow_frozen_absolute': (.0007, False, 'absolute'),
    'slow_open64': (.0007, True, 'none'),
    'slow_open64_absolute': (.0007, True, 'absolute'),
    'slow_open64_uniform_tar': (.0007, True, 'uniform_tar'),
    'slow_open64_entropy_tar': (.0007, True, 'entropy_tar'),
    'slow_open64_entropy_dggp': (.0007, True, 'entropy_dggp'),
    'standard_open64': (.007, True, 'none'),
}
DIRECTIONS = ['head_old_gold_new', 'head_new_gold_old', 'both_new']


def setup(prefix, ids, lr, opened):
    frozen = prefix['E'][:ex.N].cuda().clone()
    newp = torch.nn.Parameter(prefix['E'][ex.N:].cuda().clone())
    oldp = torch.nn.Parameter(frozen[ids].clone()) if opened else None
    optnew = torch.optim.Adam([newp], lr=lr)
    saved = next(iter(prefix['optimizer']['state'].values()))
    state = {k: v[ex.N:].cuda().clone() if k in ['exp_avg', 'exp_avg_sq'] else copy.deepcopy(v)
             for k, v in saved.items()}
    optnew.state[newp] = state
    optold = torch.optim.Adam([oldp], lr=lr) if opened else None
    R = prefix['R'].cuda().clone()
    def entity():
        old = frozen.index_copy(0, ids, oldp) if opened else frozen
        return torch.cat([old, newp])
    return newp, oldp, optnew, optold, R, entity


def packed(g, ids, opened):
    return torch.cat([g[ids], g[ex.N:]]) if opened else g[ex.N:]


def update(g, newp, oldp, optnew, optold):
    optnew.zero_grad(set_to_none=True)
    if oldp is None:
        newp.grad = g
    else:
        optold.zero_grad(set_to_none=True)
        oldp.grad, newp.grad = g[:len(oldp)], g[len(oldp):]
        optold.step()
    optnew.step()


def tar(E, teacher, ids, weights):
    return (weights * (E[ids] - teacher[ids]).square().sum(-1)).mean()


def positive_scores(E, R, triples):
    h, r, t = triples.unbind(1)
    return (E[h] * R[r] * E[t]).sum(-1)


def absolute(E, R, data, selected):
    q = torch.as_tensor(data['queries'][selected], device='cuda')
    candidates = torch.as_tensor(data['candidates'][selected], device='cuda')
    # Current query vectors must be recomputed after old entities are unfrozen.
    query = E[q[:, 0]].double() * R[q[:, 1]].double()
    return F.softplus((query[:, None] * E[candidates].double()).sum(-1)).mean()


def prepare(initial, prefix, old, new, tape, data):
    start = time.monotonic()
    endpoints = np.concatenate([new[:, 0], new[:, 2]])
    count = np.bincount(endpoints[endpoints < ex.N], minlength=ex.N)
    picked = np.lexsort((np.arange(ex.N), -count))[:64]
    ids = torch.as_tensor(picked, device='cuda')
    topo = ex.base.load('/data/lc/untzh/tadg61_20261013/topology.npz')
    weights = torch.as_tensor(topo['weights'][picked], device='cuda', dtype=prefix['E'].dtype)
    rng = np.random.default_rng(17)
    selected = np.stack([rng.choice(ex.SIZE, 32, replace=False) for _ in range(370)])
    canonical = old[old[:, 1] % 2 == 0]
    memory = canonical[rng.choice(len(canonical), 512, replace=False)]
    memory_t = torch.as_tensor(memory, device='cuda')
    memory_batches = []
    for j in range(370, 740):
        lo, hi = tape['bounds'][j:j + 2]
        p = new[tape['pos'][lo:hi]]
        endpoints_batch = np.unique(p[:, [0, 2]])
        matches = np.flatnonzero(np.isin(memory[:, 0], endpoints_batch) | np.isin(memory[:, 2], endpoints_batch))
        memory_batches.append(matches if len(matches) else np.arange(512))
    teacher = prefix['E'].cuda()
    teacher_scores = positive_scores(teacher, prefix['R'].cuda(), memory_t).detach()
    newp, oldp, on, oo, R, entity = setup(prefix, ids, .0007, True)
    bn, pn = [], []
    for j in range(370, 378):
        E = entity()
        loss = ex.base.learning(E, R, new, tape, j)
        g = packed(torch.autograd.grad(loss, E)[0], ids, True)
        update(g, newp, oldp, on, oo)
        E = entity()
        loss = ex.base.learning(E, R, new, tape, j)
        g = packed(torch.autograd.grad(loss, E)[0], ids, True)
        pen = tar(E, teacher, ids, torch.ones_like(weights))
        gp = packed(torch.autograd.grad(pen, E)[0], ids, True)
        bn.append(float(g.double().norm())); pn.append(float(gp.double().norm()))
    alpha_tar = .1 * np.sqrt(np.mean(np.square(bn))) / np.sqrt(np.mean(np.square(pn)))
    assert np.isfinite(alpha_tar) and alpha_tar > 0
    np.savez(ROOT / 'prepared.npz', opened_ids=picked, connection_counts=count[picked],
             entropy_weights=weights.cpu().numpy(), protected_answers=selected, memory=memory)
    ex.dump('preparation.json', {'seconds': time.monotonic() - start, 'tar_alpha': alpha_tar,
        'tar_probe_learning_norms': bn, 'tar_probe_penalty_norms': pn,
        'memory_size': 512, 'memory_facts_touching_opened_rows': int((np.isin(memory[:, 0], picked) | np.isin(memory[:, 2], picked)).sum()),
        'opened_ids': picked.tolist(), 'connection_counts': count[picked].tolist(),
        'entropy_weight_range': [float(weights.min()), float(weights.max())],
        'rule': '64 highest new-TRAIN endpoint counts, ID tie break; 8-step discarded TRAIN-only slow probe calibrates uniform TAR to rho=.1; shared alpha and per-step cap'})
    return ids, weights, selected, memory_t, memory_batches, teacher_scores, float(alpha_tar)


def train(name, config, prefix, new, tape, data, prepared, coefficient):
    lr, opened, mode = config
    ids, weights, selected, memory, memory_batches, teacher_scores, alpha_tar = prepared
    start = time.monotonic()
    newp, oldp, on, oo, R, entity = setup(prefix, ids, lr, opened)
    teacher = prefix['E'].cuda()
    records = []
    for j in range(370, 740):
        E = entity()
        loss = ex.base.learning(E, R, new, tape, j)
        gl = packed(torch.autograd.grad(loss, E)[0], ids, opened)
        g = gl
        penalty, alpha, ratio = 0., 0., 0.
        if mode != 'none':
            if mode == 'absolute':
                pen = absolute(E, R, data, selected[j - 370])
                coefficient_here = coefficient
            else:
                w = torch.ones_like(weights) if mode == 'uniform_tar' else weights
                pen = tar(E, teacher, ids, w)
                coefficient_here = alpha_tar
            gp = packed(torch.autograd.grad(pen, E)[0], ids, opened)
            bn, pn = float(gl.double().norm()), float(gp.double().norm())
            alpha = min(coefficient_here, .1 * bn / pn) if pn else coefficient_here
            ratio = alpha * pn / bn if bn else 0.
            g = gl + alpha * gp
            penalty = float(pen)
        changed, dot_before, dot_after, distill = False, None, None, None
        if mode == 'entropy_dggp':
            batches = memory_batches[j - 370]
            distill_loss = F.huber_loss(positive_scores(E, R, memory[batches]), teacher_scores[batches], delta=1.)
            gr = packed(torch.autograd.grad(distill_loss, E)[0], ids, opened)
            a, b = g.double(), gr.double()
            dot, norm2 = (a * b).sum(), b.square().sum()
            dot_before = float(dot)
            if dot < 0 and norm2 > 0:
                a = a - dot / norm2 * b
                g = a.to(g.dtype); changed = True
            dot_after = float((g.double() * b).sum())
            distill = float(distill_loss)
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite gradient/loss')
        update(g, newp, oldp, on, oo)
        records.append({'step': j, 'epoch': int(tape['epochs'][j]), 'learning_loss': float(loss),
            'penalty': penalty, 'alpha': alpha, 'penalty_gradient_ratio': ratio,
            'distill': distill, 'projected': changed, 'reference_inner_before': dot_before, 'reference_inner_after': dot_after})
        if j == 739 or tape['epochs'][j + 1] != tape['epochs'][j]:
            ex.progress('train', arm=name, epoch=int(tape['epochs'][j]), seconds=time.monotonic() - start)
    E = entity().detach().cpu()
    fixed = torch.ones(ex.N, dtype=torch.bool)
    if opened:
        fixed[ids.cpu()] = False
    assert torch.equal(E[:ex.N][fixed], prefix['E'][:ex.N][fixed])
    assert torch.equal(R.cpu(), prefix['R']) and torch.isfinite(E).all()
    drift = (E[:ex.N] - prefix['E'][:ex.N]).double().norm(dim=1)
    out = ROOT / 'models' / name; out.mkdir(parents=True)
    torch.save({'E': E.double(), 'R': R.cpu().double()}, out / 'epoch-20.pt')
    ex.base.write(out / 'steps.json', records)
    result = {'seconds': time.monotonic() - start, 'learning_rate': lr, 'opened_old_rows': 64 if opened else 0,
        'drift_l2': float(drift.norm()), 'max_row_drift': float(drift.max()),
        'projected_steps': sum(x['projected'] for x in records), 'steps': 370,
        'frozen_rows_bitwise_equal': True, 'new_adam_step': float(on.state[newp]['step']),
        'old_adam_step': float(oo.state[oldp]['step']) if opened else None}
    ex.base.write(out / 'training.json', result)
    return result


def drops(ranks, baseline):
    return {kind: {'eligible': int((baseline[:, col] <= 10).sum()),
                  'drops': int(((baseline[:, col] <= 10) & (ranks[:, col] > 10)).sum())}
            for col, kind in enumerate(['raw', 'filtered_train_truths'])}


@torch.no_grad()
def evaluate(old, new, A):
    start = time.monotonic()
    truth = collections.defaultdict(set)
    for h, r, t in np.concatenate([old, new]):
        truth[int(h), int(r)].add(int(t))
    valid = ex.base.load(ex.SRC / 'entity_refresh43_20261012/artifacts/data/validation.npz')
    prior = json.loads((PRIOR / 'evaluation/metrics.json').read_text())
    report = {'standard_frozen': prior['new_only'], 'standard_frozen_absolute': prior['uniform']}
    out = ROOT / 'evaluation'; out.mkdir(exist_ok=True)
    chosen = []
    for name in ARMS:
        model = torch.load(ROOT / 'models' / name / 'epoch-20.pt', map_location='cpu', weights_only=True)
        E, R = model['E'].cuda(), model['R'].cuda()
        ranks = {split: ex.base.ranks(E, R, valid[key], truth, arm=name, split=split)
                 for split, key in [('old_valid', 'old'), ('new_valid', 'new')]}
        np.savez(out / f'{name}-valid.npz', **ranks)
        report[name] = {split: ex.base.metrics(ranks[split], valid[key])
                        for split, key in [('old_valid', 'old'), ('new_valid', 'new')]}
        item = report[name]
        item['old_valid']['drops'] = drops(ranks['old_valid'], A['old_valid'])
        ratios = [item['new_valid'][d]['filtered_train_truths']['mrr'] / prior['new_only']['new_valid'][d]['filtered_train_truths']['mrr'] for d in DIRECTIONS]
        mrr = item['new_valid']['all']['filtered_train_truths']['mrr']
        oldmrr = item['old_valid']['all']['filtered_train_truths']['mrr']
        oldbase = prior['new_only']['old_valid']['all']['filtered_train_truths']['mrr']
        passed = min(ratios) >= .98 and item['old_valid']['drops']['filtered_train_truths']['drops'] <= 5 and oldmrr >= .99 * oldbase
        item['development_screen_pass'] = passed
        item['directional_ratios_vs_standard_ordinary'] = ratios
        if passed:
            chosen.append(name)
        ex.progress('valid_screen', arm=name, new_mrr=mrr, old_mrr=oldmrr,
                    drops=item['old_valid']['drops']['filtered_train_truths']['drops'], ratios=ratios, passed=passed)
    valid_seconds = time.monotonic() - start
    for name in chosen:
        model = torch.load(ROOT / 'models' / name / 'epoch-20.pt', map_location='cpu', weights_only=True)
        a = ex.base.ranks(model['E'].cuda(), model['R'].cuda(), old, truth, arm=name, split='old_train')
        np.savez(out / f'{name}-train.npz', old_train=a)
        report[name]['old_train'] = ex.base.metrics(a, old)
        report[name]['old_train']['drops'] = drops(a, A['old_train'])
        ex.progress('full_train_evaluation', arm=name)
    ex.dump('evaluation/metrics.json', report)
    return report, {'seconds': time.monotonic() - start, 'valid_seconds': valid_seconds, 'full_train_arms': chosen}


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic(); ex.progress('starting')
    initial = torch.load(ex.SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt', map_location='cpu', weights_only=True)
    prefix = torch.load(PRIOR / 'cache/prefix10.pt', map_location='cpu', weights_only=True)
    d = ex.base.load(ex.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old, new = d['old'], d['new']
    tape = ex.base.load(ex.SRC / 'matched_protection52_20261013/artifacts/tape/learning.npz')
    data = ex.base.load(PRIOR / 'cache/cohort.npz')
    prepared = prepare(initial, prefix, old, new, tape, data)
    coefficient = json.loads((PRIOR / 'coefficient.json').read_text())['lambda']
    costs = {'training': {}, 'shared_prefix_cache': str(PRIOR / 'cache/prefix10.pt'),
             'preparation': json.loads((ROOT / 'preparation.json').read_text())}
    ex.dump('environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(), 'seed': 17})
    for name, config in ARMS.items():
        costs['training'][name] = train(name, config, prefix, new, tape, data, prepared, coefficient)
    A = ex.base.load(ex.SRC / 'replay_strength53_20261013/artifacts/evaluation/A.npz')
    quality, evaluation = evaluate(old, new, A)
    costs['evaluation'] = evaluation; costs['total_seconds'] = time.monotonic() - started
    report_results(quality, costs)
    ex.progress('complete', seconds=time.monotonic() - started)


def report_results(quality, costs):
    ex.dump('costs.json', costs)
    lines = ['# 062: limited old-row unfreezing', '', 'Seed17 exposed-VALID development screen; adaptations, not native TADG.', '',
             '| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old-new ratio | New-old ratio | New-new ratio | Screen |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    baseline = quality['standard_frozen']['new_valid']
    for name, item in quality.items():
        nv = item['new_valid']
        ratios = [nv[d]['filtered_train_truths']['mrr'] / baseline[d]['filtered_train_truths']['mrr'] for d in DIRECTIONS]
        lines.append(f"| {name} | {nv['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['all']['filtered_train_truths']['mrr']:.9f} | {item['old_valid']['drops']['filtered_train_truths']['drops']} | {ratios[0]:.6f} | {ratios[1]:.6f} | {ratios[2]:.6f} | {item.get('development_screen_pass', 'reused')} |")
    lines += ['', 'Only screened configurations receive full TRAIN evaluation; absent TRAIN metrics mean uncomputed.',
              'New Adam moments/step370 retained; old opened rows fresh Adam step0. Relations and other old rows remain fixed.',
              'All new runs share the 10-epoch prefix. Lower-rate comparisons use .0007 matched controls; standard pair uses .007.',
              'No TEST, independent confirmation, time limit, audit, SHA gates, or archive.']
    (ROOT / 'results.md').write_text('\n'.join(lines) + '\n')



if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        ex.progress('failed', error=repr(exc)); raise
