import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/complex_transfer118_20261014')
import experiment118 as cx
base, core = cx.base, cx.core
ROOT = Path(__file__).resolve().parent
PARENTS = [17, 'WN_ComplEx']
STEPS = 64
MAX_REFERENCE = 2048
RIDGE = .01

def status(stage, **kw):
    base.write(ROOT / 'status.json', dict(stage=stage, pid=os.getpid(), **kw))

def query(initial, facts, complex_model):
    E, R = initial['E'].cuda().double(), initial['R'].cuda().double()
    b = torch.as_tensor(facts, device='cuda')
    q = cx.tail_query(E[b[:, 0]], R[b[:, 1]]) if complex_model else E[b[:, 0]] * R[b[:, 1]]
    return q

def setup(parent):
    fb = isinstance(parent, int)
    n, total = (11258, 13134) if fb else (24567, 28660)
    src = Path('/data/lc/untzh') / ('increment87_20261014' if fb else 'complex_transfer118_20261014')
    path = Path('/data/lc/untzh/train_init121_20261014/seed17/models/random_half/initial.pt') if fb else Path('/data/lc/untzh/wn_init122_20261014/ComplEx/models/random_half/initial.pt')
    initial = torch.load(path, map_location='cpu', weights_only=True)
    queries = base.load(src / 'evaluation/queries.npz')
    if fb:
        queries['old_valid'] = base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid']
        z = base.load(src / 'cache/train.npz')
        truths = core.truthmap(z['old'], np.concatenate([z['new'], z['extra']]))
    else:
        z = base.load('/data/lc/untzh/wn_transfer112_20261014/cache/train.npz')
        truths = core.truthmap(z['old'], z['new'])
    predictions = {s: base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-{s}-half-prediction.npz') for s in ['old_train', 'old_valid']}
    split = base.load(f'/data/lc/untzh/admission_action128_20261014/cache/{parent}-split.npz')
    return n, total, src, initial, queries, truths, predictions, split

def loss_report(q, a, x):
    gap = (q @ x.T - a[:, None]).clamp_min(0)
    return dict(positive_gap_mean=float(gap.mean()), squared_gap_mean=float(gap.square().mean()),
                satisfied_fraction=float((gap == 0).double().mean()),
                violated_queries=int((gap.max(1).values > 1e-12).sum()))

def fit(initial, n, facts, prediction, mask, complex_model, out):
    tick = time.monotonic()
    candidates = np.flatnonzero(mask)
    margin = prediction['gold'] / np.maximum(prediction['querynorm'], 1e-30)
    order = np.argsort(margin[candidates], kind='stable')[:MAX_REFERENCE]
    positions = candidates[order]
    reference = facts[prediction['ids'][positions]]
    q = query(initial, reference, complex_model)
    norm = q.norm(dim=1)
    assert bool((norm > 0).all())
    q /= norm[:, None]
    a = torch.as_tensor(prediction['gold'][positions], device='cuda') / norm
    x = initial['E'][n:].cuda().double().clone()
    radius = x.norm(dim=1, keepdim=True)
    before = loss_report(q, a, x)
    spectral = float(torch.linalg.eigvalsh(q.T @ q / len(q)).max())
    lr = 1 / (2 * spectral + 2 * RIDGE)
    for step in range(STEPS):
        gap = (q @ x.T - a[:, None]).clamp_min(0)
        gradient = 2 / len(q) * (gap.T @ q) + 2 * RIDGE * x
        x -= lr * gradient
        x *= (radius / x.norm(dim=1, keepdim=True).clamp_min(1e-30)).clamp_max(1)
        assert bool(torch.isfinite(x).all())
    after = loss_report(q, a, x)
    r = x.norm(dim=1, keepdim=True)
    direction = initial['E'][n:].cuda().double()
    random_match = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-30) * r
    variants = dict(constrained=x, normmatched_random=random_match, reverse=-x)
    for arm, v in variants.items():
        state = dict(E=initial['E'].clone(), R=initial['R'])
        state['E'][n:] = v.cpu().float()
        torch.save(state, out / f'{arm}-initial.pt')
    np.savez(out / 'reference-selection.npz', positions=positions, query_ids=prediction['ids'][positions], reference=reference)
    report = dict(seconds=time.monotonic()-tick, selected=len(q), available=int(mask.sum()),
                  steps=STEPS, step_size=lr, ridge=RIDGE, mean_radius_before=float(radius.mean()),
                  mean_radius_after=float(r.mean()), max_radius_increase=float((r-radius).max()), before=before, after=after,
                  selected_negative_gold=int((a < 0).sum()), norm_match_max_error=float((random_match.norm(dim=1)-r[:,0]).abs().max()))
    base.write(out / 'fit.json', report)
    return report

@torch.no_grad()
def expansion(initial, n, facts, pred, truths, complex_model):
    tick = time.monotonic()
    E, R = initial['E'].cuda().double(), initial['R'].cuda().double()
    rows = E[n:]
    all_ranks = []
    for lo in range(0, len(pred['ids']), 256):
        ids = pred['ids'][lo:lo+256]
        b = facts[ids]
        a = torch.as_tensor(b, device='cuda')
        q = cx.tail_query(E[a[:,0]], R[a[:,1]]) if complex_model else E[a[:,0]] * R[a[:,1]]
        ss = q @ rows.T
        gold = torch.as_tensor(pred['gold'][lo:lo+len(b)], device='cuda')
        for j, (h, r, g) in enumerate(b):
            forbidden = [v-n for v in truths[int(h),int(r)] if v >= n]
            if forbidden:
                ss[j, forbidden] = -torch.inf
        added = (ss > gold[:,None]).sum(1).cpu().numpy()
        all_ranks.append(pred['oldrank'][lo:lo+len(b)] + added)
    return np.concatenate(all_ranks), dict(seconds=time.monotonic()-tick, queries=len(pred['ids']))

def describe(rank, mask):
    return dict(eligible=int(mask.sum()), drops=int(((rank>10)&mask).sum()),
                eligible_mrr=float((1/rank[mask]).mean()) if mask.any() else None)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick = time.monotonic()
    status('prepare')
    bundles=[]
    reports={}
    for parent in PARENTS:
        n,total,src,initial,queries,truths,predictions,split=setup(parent)
        fits={}
        for condition in ['train_only','historical_reference']:
            out=ROOT/str(parent)/condition
            out.mkdir(parents=True, exist_ok=True)
            p=predictions['old_train' if condition=='train_only' else 'old_valid']
            mask=np.ones(len(p['ids']),bool) if condition=='train_only' else split['calibration'][p['ids']]
            status('fit',parent=str(parent),condition=condition)
            fits[condition]=fit(initial,n,queries['old_train' if condition=='train_only' else 'old_valid'],p,mask,parent=='WN_ComplEx',out)
            print('FIT130',parent,condition,fits[condition],flush=True)
        bundles.append((parent,n,initial,queries,truths,predictions,split,fits))
    base.write(ROOT/'committed-initializations.json',dict(parents=[str(p) for p in PARENTS],initializations=12,
        current_endpoint_evaluation_not_used=True,incremental_training=False))
    # Every initialization committed before held-group initial rank read or scoring.
    for parent,n,initial,queries,truths,predictions,split,fits in bundles:
        p=predictions['old_valid']
        cal=split['calibration'][p['ids']]
        held=split['evaluation'][p['ids']]
        report=dict(fits=fits,conditions={},fixed_controls={})
        evaluation=ROOT/str(parent)/'evaluation'
        evaluation.mkdir(exist_ok=True)
        for condition in ['train_only','historical_reference']:
            report['conditions'][condition]={}
            for arm in ['constrained','normmatched_random','reverse']:
                status('evaluate',parent=str(parent),condition=condition,arm=arm)
                state=torch.load(ROOT/str(parent)/condition/f'{arm}-initial.pt',map_location='cpu',weights_only=True)
                rank,cost=expansion(state,n,queries['old_valid'],p,truths,parent=='WN_ComplEx')
                np.savez(evaluation/f'{condition}-{arm}.npz',ids=p['ids'],rank=rank)
                m=dict(calibration=describe(rank,cal),held_group=describe(rank,held),cost=cost)
                report['conditions'][condition][arm]=m
                print('INITIAL130',parent,condition,arm,m,flush=True)
        for mode in ['zero','random_half']:
            folder=Path('/data/lc/untzh/train_init121_20261014/seed17') if parent==17 else Path('/data/lc/untzh/wn_init122_20261014/ComplEx')
            cached=base.load(folder/'evaluation'/f'{mode}-old_valid-phases.npz')['expanded_initial'][p['ids'],1]
            report['fixed_controls'][mode]=dict(calibration=describe(cached,cal),held_group=describe(cached,held))
        reports[str(parent)]=report
        base.write(ROOT/str(parent)/'summary.json',report)
        base.write(ROOT/'summary-running.json',dict(parents=reports))
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_initializations=12,
        new_incremental_models=0,gaussian_probability_not_applicable_after_optimization=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
