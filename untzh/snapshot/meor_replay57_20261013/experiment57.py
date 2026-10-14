"""MEOR formula adaptation and paired replay comparisons using prepared inputs."""
import collections
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import base_experiment as b

ROOT = Path(__file__).resolve().parent
PREPARED = Path('/data/lc/untzh/signals56_prepare_optimization_20261013')
PRIOR = Path('/data/lc/untzh/signals56_20261013')
ARMS = ['replay', 'meor_stepcap', 'absolute_stepcap', 'replay_plus_meor_stepcap']
RHO = .1
b.ROOT, b.ARMS = ROOT, ARMS


def cached(j):
    return {k: torch.as_tensor(v, device=b.DEVICE) for k, v in
            b.load(PREPARED / 'cache' / f'{j:04d}.npz').items()}


def prepare():
    started = time.monotonic()
    saved = json.loads((PREPARED / 'prepared-contract.json').read_text())
    for p, st in saved['readonly_inputs'].items():
        current = Path(p).stat()
        if {'bytes': current.st_size, 'mtime_ns': current.st_mtime_ns} != st:
            raise ValueError('Preparation input changed: ' + p)
    expected = {'old_count':7505,'entity_count':11258,'protection_seed':550017,
                'steps':740,'answers_per_step':32,'reference_draws':4,'scale_count':256,
                'scale_eta':1.4826,'scale_epsilon':1e-8,'scale_adequate':32}
    if any(saved[k] != v for k, v in expected.items()):
        raise ValueError('Preparation recipe changed')
    if any(not (PREPARED / 'cache' / f'{j:04d}.npz').is_file() for j in range(740)):
        raise ValueError('Incomplete preparation')
    previous = json.loads((PREPARED / 'prepare.json').read_text())
    result = {'reused': True, 'source':str(PREPARED), 'steps':740,
              'reuse_check_seconds':time.monotonic()-started,
              'historical_preparation_seconds':previous['seconds'],
              'historical_components':previous['components'], 'cache_bytes':previous['cache_bytes'],
              'note':'Historical shared random/structural preparation; not standalone MEOR deployment cost.'}
    b.write(ROOT / 'prepare.json', result)
    return result


def replay_loss(E, R, old, tape, j):
    lo, hi = tape['bounds'][j:j+2]
    q = torch.as_tensor(old[tape['old_ids'][lo:hi]], device=b.DEVICE, dtype=torch.long)
    candidates = torch.as_tensor(tape['old_candidates'][lo:hi], device=b.DEVICE, dtype=torch.long)
    vectors = E[q[:, 0]] * R[q[:, 1]]
    return F.softplus((vectors[:, None, :] * E[candidates]).sum(-1)).mean()


def host(E, R, old, new, tape, j, arm):
    base = b.learning(E, R, new, tape, j)
    rep = replay_loss(E,R,old,tape,j) if arm in ('replay','replay_plus_meor_stepcap') else base.new_zeros(())
    return base + rep, base, rep


def regularizer(E, data, arm):
    return b.penalty(E, data, 'absolute' if arm == 'absolute_stepcap' else 'structural')


def norm(g):
    value = float(torch.linalg.vector_norm(g[b.N:].double()))
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite gradient norm')
    return value


def calibrate(initial, old, new, tape):
    start = time.monotonic()
    E = torch.nn.Parameter(initial['E'].to(b.DEVICE).clone())
    R = initial['R'].to(b.DEVICE).clone()
    result = {}
    for arm in ARMS[1:]:
        hn, pn = [], []
        for j in range(8):
            loss, _, _ = host(E,R,old,new,tape,j,arm)
            hn.append(norm(torch.autograd.grad(loss,E)[0]))
            pn.append(norm(torch.autograd.grad(regularizer(E,cached(j),arm),E)[0]))
        result[arm] = {**b.coefficient(hn,pn,rho=RHO), 'host_norms':hn,'protection_norms':pn}
        print('CALIBRATE',arm,result[arm]['lambda'],flush=True)
    b.write(ROOT / 'coefficients.json', result)
    return result, time.monotonic()-start


def train(initial, old, new, tape, arm, lam):
    start = time.monotonic()
    out = ROOT / 'models' / arm
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    E = torch.nn.Parameter(initial['E'].to(b.DEVICE).clone())
    R = initial['R'].to(b.DEVICE).clone()
    optimizer = torch.optim.Adam([E],lr=.007)
    records = []
    for j, ep in enumerate(tape['epochs']):
        loss, base, replay = host(E,R,old,new,tape,j,arm)
        gh = torch.autograd.grad(loss,E)[0]
        hn = norm(gh)
        penalty = loss.new_zeros(())
        pn, alpha, cosine, active = 0.,0.,None,0
        if arm != 'replay':
            data = cached(j)
            penalty = regularizer(E,data,arm)
            gp = torch.autograd.grad(penalty,E)[0]
            pn = norm(gp)
            alpha = min(lam, RHO*hn/pn) if pn else 0.
            if hn and pn:
                cosine = float((gh[b.N:].double()*gp[b.N:].double()).sum()/(hn*pn))
            if arm != 'absolute_stepcap':
                with torch.no_grad():
                    u = b.aggregate(data['q'] @ E[b.N:].double().T,data['gold'],data['scale'],data['mask'])
                    active = int((u > data['structural_mean']).sum())
            gh = gh + alpha*gp
        if not torch.isfinite(loss) or not torch.isfinite(penalty) or not torch.isfinite(gh).all():
            raise FloatingPointError('Nonfinite training state')
        gh[:b.N] = 0
        optimizer.zero_grad(set_to_none=True)
        E.grad = gh
        optimizer.step()
        records.append({'step':j,'epoch':int(ep),'new_learning_loss':float(base.detach()),
                        'replay_loss':float(replay.detach()),'host_loss':float(loss.detach()),
                        'meor_or_absolute_loss':float(penalty.detach()),'host_gradient_norm':hn,
                        'protection_gradient_norm':pn,'fixed_lambda':lam,'effective_alpha':alpha,
                        'uncapped_gradient_ratio':lam*pn/hn if hn else None,
                        'actual_gradient_ratio':alpha*pn/hn if hn else None,
                        'host_protection_cosine':cosine,'active_occurrences':active,
                        'capped':alpha < lam if arm != 'replay' else False,
                        'weighted_protection_loss':alpha*float(penalty.detach())})
        if j == len(tape['epochs'])-1 or tape['epochs'][j+1] != ep:
            if not torch.equal(E[:b.N].detach().cpu(),initial['E'][:b.N]) or not torch.equal(R.cpu(),initial['R']):
                raise ValueError('Frozen parameters changed')
            print('TRAIN',arm,int(ep),records[-1]['host_loss'],records[-1]['actual_gradient_ratio'],flush=True)
            b.write(out / 'losses.json',records)
            b.write(ROOT / 'status.json',{'stage':'train','arm':arm,'epoch':int(ep),'step':j+1,'pid':os.getpid()})
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    result = {'seconds':time.monotonic()-start,'steps':len(records),'fixed_lambda':lam,
              'capped_steps':sum(x['capped'] for x in records),
              'max_actual_gradient_ratio':max((x['actual_gradient_ratio'] or 0.) for x in records),
              'max_uncapped_gradient_ratio':max((x['uncapped_gradient_ratio'] or 0.) for x in records),
              'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'old_E_R_bitwise_frozen':True}
    if arm == 'replay':
        historical = torch.load(b.SRC/'matched_protection52_20261013/artifacts/train/absolute_replay/epoch-20.pt',map_location='cpu',weights_only=True)
        result['historical_replay_E_equal'] = torch.equal(E.detach().cpu().double(),historical['E'])
        result['historical_replay_R_equal'] = torch.equal(R.cpu().double(),historical['R'])
    b.write(out/'training.json',result)
    return result


def paired_interval(delta, queries, seed):
    _, inverse = np.unique(queries[:,:2],axis=0,return_inverse=True)
    totals = np.bincount(inverse,weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(5000):
        take = rng.integers(0,len(counts),len(counts))
        values.append(float(totals[take].sum()/counts[take].sum()))
    return {'mean_difference':float(np.mean(delta)), 'query_clusters':len(counts),
            'descriptive_95pct_query_bootstrap':np.quantile(values,[.025,.975]).tolist()}


def compare(report):
    queries = b.load(ROOT/'evaluation/queries.npz')
    prior_eval = b.SRC/'replay_strength53_20261013/artifacts/evaluation'
    states = {arm:b.load(ROOT/'evaluation'/f'{arm}.npz') for arm in ARMS}
    states['new_only'] = b.load(prior_eval/'new_only-e20.npz')
    A = b.load(prior_eval/'A.npz')
    pairs = [('meor_stepcap','new_only'),('absolute_stepcap','new_only'),
             ('meor_stepcap','absolute_stepcap'),('replay_plus_meor_stepcap','replay')]
    result = {}
    directions = ['head_old_gold_new','head_new_gold_old','both_new']
    for index,(candidate,baseline) in enumerate(pairs):
        m, ref = report[candidate], report[baseline]
        ratios = {d:m['new_valid'][d]['filtered_train_truths']['mrr']/ref['new_valid'][d]['filtered_train_truths']['mrr'] for d in ['all']+directions}
        hits = {d:m['new_valid'][d]['filtered_train_truths']['hits10']/ref['new_valid'][d]['filtered_train_truths']['hits10'] for d in directions}
        oldratio = m['old_valid']['all']['filtered_train_truths']['mrr']/ref['old_valid']['all']['filtered_train_truths']['mrr']
        drops = m['old_valid']['drops']['filtered_train_truths']['drops']
        refdrops = ref['old_valid']['drops']['filtered_train_truths']['drops']
        gate = {'new_mrr_all_and_directions_at_least_98pct':all(v>=.98 for v in ratios.values()),
                'direction_hits10_at_least_95pct':all(v>=.95 for v in hits.values()),
                'old_valid_mrr_at_least_98pct':oldratio>=.98,
                'old_valid_drops_reduce_at_least_25pct':drops<=.75*refdrops,
                'old_train_drops_no_worse':m['old_train']['drops']['filtered_train_truths']['drops']<=ref['old_train']['drops']['filtered_train_truths']['drops']}
        intervals = {}
        for split in ['new_valid','old_valid']:
            q = queries[split]
            ra, rb = states[candidate][split][:,1],states[baseline][split][:,1]
            intervals[split+'_mrr'] = paired_interval(1/ra-1/rb,q,570017+index)
            if split=='new_valid':
                masks = {'head_old_gold_new':(q[:,0]<b.N)&(q[:,2]>=b.N),
                         'head_new_gold_old':(q[:,0]>=b.N)&(q[:,2]<b.N),
                         'both_new':(q[:,0]>=b.N)&(q[:,2]>=b.N)}
                for d,mask in masks.items():
                    intervals[d+'_mrr'] = paired_interval((1/ra-1/rb)[mask],q[mask],570117+index)
            else:
                eligible = A[split][:,1]<=10
                intervals['old_valid_drop_rate'] = paired_interval(((ra>10).astype(float)-(rb>10).astype(float))[eligible],q[eligible],570217+index)
        result[candidate+'_vs_'+baseline] = {'new_filtered_mrr_ratios':ratios,
            'direction_hits10_ratios':hits,'old_valid_mrr_ratio':oldratio,
            'old_valid_drops':[drops,refdrops],'quality_gates':gate,'quality_pass':all(gate.values()),
            'paired_effects':intervals}
    b.write(ROOT/'comparisons.json',result)
    return result


def summarize(report,costs,comparisons):
    baseline = report['new_only']['new_valid']
    lines = ['# 057 MEOR adaptation versus replay','',
             'Four fresh paired arms, seed17, snapshot0→1, 20 epochs/740 steps. '
             'The MEOR aggregate and reference mean follow Equations12–15; stepcap is our explicit adaptation, not the author coefficient rule. '
             'Filtering uses cumulative TRAIN truths only; exposed VALID development evaluation, no TEST.','',
             '| Arm | New filtered MRR | Old head → new gold ratio to new_only | New head → old gold ratio | Both new ratio | Old VALID drops | Old TRAIN drops | Old VALID MRR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in report.items():
        nv = m['new_valid']
        ratios = [nv[d]['filtered_train_truths']['mrr']/baseline[d]['filtered_train_truths']['mrr'] for d in ['head_old_gold_new','head_new_gold_old','both_new']]
        lines.append(f"| {arm} | {nv['all']['filtered_train_truths']['mrr']:.9f} | {ratios[0]:.6f} | {ratios[1]:.6f} | {ratios[2]:.6f} | {m['old_valid']['drops']['filtered_train_truths']['drops']} | {m['old_train']['drops']['filtered_train_truths']['drops']} | {m['old_valid']['all']['filtered_train_truths']['mrr']:.9f} |")
    lines += ['','Fresh replay reproduces the historical weight1 sampled softplus replay objective. '
              'replay_plus_meor_stepcap adds MEOR to that identical host, with the same replay tape. '
              'meor_stepcap and absolute_stepcap share the new-only host, complete candidate cache, and cap rule. '
              'The paper calls its zero-MEOR host Replay; project historical replay includes a separate sampled negative penalty. '
              'These names must not be treated as interchangeable.','',
              'Fixed-MEOR (056 structural), fixed-full-pool absolute and replay .1/.3/1 are reused historical context, not independent repeats.','',
              '## Fixed quality criteria','']
    for name,comparison in comparisons.items():
        failed = [k for k,v in comparison['quality_gates'].items() if not v]
        lines.append(f"- {name}: {'PASS' if comparison['quality_pass'] else 'FAIL'}; failed={failed}")
    lines += ['','95% paired query bootstrap intervals are descriptive single-seed summaries; shared entities and multiple comparisons limit inference. '
              'Full metrics, individual ranks and gradient trajectories are retained. No tuning follows this comparison.','',
              '## Costs','',f"Actual new run total: {costs['total_seconds']:.3f}s; reused preparation historical cost: {costs['prepare']['historical_preparation_seconds']:.3f}s. "
              'Shared reference preparation includes random references and is not a standalone method cost. No independent audit/archive/hard timeout.']
    (ROOT/'results.md').write_text('\n'.join(lines)+'\n')
    b.write(ROOT/'costs.json',costs)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if (ROOT/'status.json').exists():
        raise ValueError('Existing run; no automatic restart')
    start = time.monotonic()
    b.write(ROOT/'status.json',{'stage':'prepare','pid':os.getpid()})
    costs = {'prepare':prepare()}
    initial = torch.load(b.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    data = b.load(b.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old,new = data['old'],data['new']
    tape = b.load(b.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tape.update(b.load(b.SRC/'matched_protection52_20261013/artifacts/tape/protection.npz'))
    assert len(tape['epochs']) == 740
    b.write(ROOT/'status.json',{'stage':'calibrate','pid':os.getpid()})
    coeff,seconds = calibrate(initial,old,new,tape)
    costs['calibration_seconds'] = seconds
    costs['train'] = {}
    for arm in ARMS:
        costs['train'][arm] = train(initial,old,new,tape,arm,0. if arm=='replay' else coeff[arm]['lambda'])
    report,seconds = b.evaluate(initial,old,new)
    costs['evaluation_seconds'] = seconds
    prior_metrics = json.loads((PRIOR/'evaluation/metrics.json').read_text())
    report['meor_fixed_056'] = prior_metrics['structural']
    report['absolute_fixed_056'] = prior_metrics['absolute']
    b.write(ROOT/'evaluation/metrics.json',report)
    comparisons = compare(report)
    costs['total_seconds'] = time.monotonic()-start
    summarize(report,costs,comparisons)
    b.write(ROOT/'status.json',{'stage':'complete','pid':os.getpid(),'seconds':time.monotonic()-start})
    print('COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        b.write(ROOT/'failure.json',{'error':repr(exc),'pid':os.getpid()})
        raise
