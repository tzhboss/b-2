import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lc/untzh/risk_increment88_20261014')
ROUTE = Path('/data/lc/untzh/anchor_route97_20261014')
CONTROL = Path('/data/lc/untzh/route_strength98_20261014')
N, TOTAL = 11258, 13134


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def dump(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def pair(a, b, teacher=None, mask=None):
    if mask is None:
        mask = np.ones(len(a), bool)
    eligible = mask if teacher is None else mask & (teacher <= 10)
    return dict(answers=int(mask.sum()), eligible=int(eligible.sum()),
                a_drops=int((eligible & (a>10)).sum()), b_drops=int((eligible & (b>10)).sum()),
                a_rescues_b=int((eligible & (a<=10) & (b>10)).sum()),
                a_harms_b=int((eligible & (a>10) & (b<=10)).sum()),
                rr_improved=int((mask & (a<b)).sum()), rr_worse=int((mask & (a>b)).sum()),
                mrr_delta_a_minus_b=float(np.mean(1/a[mask]-1/b[mask])) if mask.any() else None)


def paired():
    tick = time.monotonic()
    c = load(SOURCE/'cache/cohort.npz')
    qs = load(SOURCE/'evaluation/queries.npz')
    teacher = load(SOURCE/'evaluation/teacher.npz')
    arms = {name:load(root/'evaluation'/f'{name}.npz') for name, root in
            [('anchor_tree',ROUTE), ('anchor_pressure128',ROUTE), ('anchor_random_pool',ROUTE),
             ('uniform_tree_norm',CONTROL), ('random_tree_norm',CONTROL)]}
    exposures = {}
    for arm in ['anchor_tree','anchor_pressure128','anchor_random_pool']:
        steps = json.loads((ROUTE/'models'/arm/'steps.json').read_text())
        counts = np.zeros(len(c['ids']), np.int64)
        for s in steps:
            np.add.at(counts, s['selected'], 1)
        exposures[arm] = counts
    seen = np.zeros(len(qs['old_train']), bool)
    seen[c['ids'][exposures['anchor_tree']>0]] = True
    cohort = np.zeros_like(seen)
    cohort[c['ids']] = True
    query_keys = {tuple(row[:2]) for row in qs['old_train'][seen]}
    query_seen = np.array([tuple(row[:2]) in query_keys for row in qs['old_train']])
    masks = dict(all=np.ones_like(seen), snapshot0=np.arange(len(seen))<223284,
                 snapshot1=np.arange(len(seen))>=223284, cohort=cohort, outside_cohort=~cohort,
                 tree_selected_answer=seen, cohort_not_selected=cohort & ~seen,
                 tree_selected_query=query_seen, other_queries=~query_seen)
    report = dict(source='saved rank comparison, no neural scoring or retraining', comparisons={})
    for baseline in ['uniform_tree_norm','random_tree_norm','anchor_random_pool']:
        splits = {}
        for split, q in qs.items():
            ta = teacher[split][:,1] if split in teacher else None
            a, b = arms['anchor_tree'][split][:,1], arms[baseline][split][:,1]
            splits[split] = {'all':pair(a,b,ta)}
            if split == 'old_train':
                splits[split].update({name:pair(a,b,ta,m) for name,m in masks.items()})
            elif split == 'new_valid':
                dirs = dict(old_to_new=(q[:,0]<N)&(q[:,2]>=N), new_to_old=(q[:,0]>=N)&(q[:,2]<N), both_new=(q[:,0]>=N)&(q[:,2]>=N))
                splits[split].update({name:pair(a,b,None,m) for name,m in dirs.items()})
        report['comparisons'][baseline] = splits
    report['selections'] = {}
    for name, counts in exposures.items():
        p = counts/counts.sum()
        report['selections'][name] = dict(draws=int(counts.sum()), unique=int((counts>0).sum()),
                                         effective_exposure=float(1/(p*p).sum()), max_exposure=int(counts.max()))
    report['seconds'] = time.monotonic()-tick
    dump('paired.json',report)
    print('PAIRED', report['seconds'], report['comparisons']['uniform_tree_norm']['old_train']['all'], flush=True)
    return c, arms, exposures


@torch.no_grad()
def geometry(c, arms, exposures):
    tick = time.monotonic()
    space = torch.load('/data/lc/untzh/query_space92_20261014/cache/query-space.pt', map_location='cpu', weights_only=True)
    q = torch.as_tensor(c['q'],device='cuda',dtype=torch.float64)
    qnorm = q.square().sum(-1).clamp_min(1e-30)
    fractions = torch.ones((len(q),TOTAL-N),device='cuda',dtype=torch.float32)
    max_violation = 0.
    for group in space['groups']:
        ids, basis = group['indices'].cuda(), group['basis'].cuda()
        for lo in range(0,len(ids),32):
            ix, B = ids[lo:lo+32], basis[lo:lo+32]
            rank = B.shape[-1]
            coefficients = (q @ B.permute(1,0,2).reshape(200,-1)).reshape(len(q),len(ix),rank)
            raw = 1-coefficients.square().sum(-1)/qnorm[:,None]
            max_violation = max(max_violation,float((-raw).clamp_min(0).max()))
            fractions[:,ix] = raw.clamp(0,1).float()
    candidates = torch.as_tensor(c['candidates']-N,device='cuda')
    sampled = fractions.gather(1,candidates)
    prefix = torch.load(SOURCE/'cache/prefix10.pt',map_location='cpu',weights_only=True)['E'].cuda().double()
    sample_scores = (q[:,None,:]*prefix[N:][candidates]).sum(-1)
    top = sample_scores.argmax(-1)
    top_fraction = sampled[torch.arange(len(q),device='cuda'),top].cpu().numpy()
    mean_fraction = sampled.mean(-1).cpu().numpy()
    qvals = sampled.cpu().numpy()
    (ROOT/'cache').mkdir()
    np.savez(ROOT/'cache/actionability.npz', fraction=qvals, top_fraction=top_fraction,
             mean_fraction=mean_fraction, candidate_top=top.cpu().numpy(), ids=c['ids'])
    torch.cuda.synchronize()
    # Only TRAIN outcomes enter the diagnostic below; VALID is not used to choose a rule.
    teacher = load(SOURCE/'evaluation/teacher.npz')['old_train'][:,1]
    a = arms['anchor_tree']['old_train'][:,1][c['ids']]
    b = arms['uniform_tree_norm']['old_train'][:,1][c['ids']]
    ta = teacher[c['ids']]
    pools = {name:np.array(json.loads((ROUTE/'models'/name/'decisions.json').read_text())[0]['pool']) for name in exposures}
    report = dict(definition='||P_j q||^2 / ||q||^2 on fixed TRAIN query space, sampled legal candidates128; theoretical first-order score direction only',
                  feature_epoch=10, max_numerical_violation=max_violation, sample_bytes=qvals.nbytes,
                  pair_quantiles=np.quantile(qvals,[0,.01,.05,.25,.5,.75,.95,1]).tolist(),
                  top_quantiles=np.quantile(top_fraction,[0,.01,.05,.25,.5,.75,.95,1]).tolist(),
                  low_top_fraction_counts={str(v):int((top_fraction<v).sum()) for v in [.1,.25,.5,.75,.9]}, pools={}, bins={})
    for name, ix in pools.items():
        report['pools'][name] = dict(answers=len(ix), top_fraction_mean=float(top_fraction[ix].mean()),
                                    top_fraction_quantiles=np.quantile(top_fraction[ix],[0,.25,.5,.75,1]).tolist(),
                                    selected_unique=int((exposures[name]>0).sum()),
                                    weighted_top_fraction=float(np.average(top_fraction,weights=exposures[name])))
    for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,.9),(.9,1.000001)]:
        mask=(top_fraction>=lo)&(top_fraction<hi)
        report['bins'][f'{lo}-{hi}'] = pair(a,b,ta,mask)
    report['seconds']=time.monotonic()-tick
    dump('geometry.json',report)
    print('GEOMETRY',json.dumps(report),flush=True)
    return report


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    tick=time.monotonic()
    dump('status.json',dict(stage='paired',pid=os.getpid()))
    c, arms, exposures = paired()
    dump('status.json',dict(stage='geometry',pid=os.getpid()))
    geometry(c,arms,exposures)
    dump('status.json',dict(stage='complete',pid=os.getpid(),seconds=time.monotonic()-tick))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('status.json',dict(stage='failed',pid=os.getpid(),error=repr(exc)))
        raise
