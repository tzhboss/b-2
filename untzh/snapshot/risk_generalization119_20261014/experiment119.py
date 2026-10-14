import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
import joblib
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
sys.path.insert(0, '/data/lc/untzh/complex_transfer118_20261014')
import experiment118 as cx
base, core = cx.base, cx.core
ROOT = Path(__file__).resolve().parent

def status(stage, **kw):
    base.write(ROOT/'status.json', dict(stage=stage, pid=os.getpid(), **kw))

def sample(queries, truths, n, total, seed):
    rng = np.random.default_rng(seed)
    candidates = rng.integers(n, total, size=(len(queries), 128))
    oldsample = rng.integers(n, size=(len(queries), 64))
    for i, (h, r, g) in enumerate(queries):
        known = list(truths[int(h), int(r)])
        bad = np.isin(candidates[i], known)
        while bad.any():
            candidates[i, bad] = rng.integers(n, total, size=int(bad.sum()))
            bad = np.isin(candidates[i], known)
    return candidates, oldsample

@torch.no_grad()
def features(initial, prefix, queries, teacher, gold_cache, train, truths,
             candidates, oldsample, total, complex_model, delta, epoch_steps):
    torch.cuda.synchronize(); tick = time.monotonic()
    e, r = initial['E'].cuda().double(), initial['R'].cuda().double()
    pe = prefix['E'].cuda().double()
    delta = delta.cuda().double()
    q = cx.tail_query(e[queries[:, 0]], r[queries[:, 1]]) if complex_model else e[queries[:, 0]] * r[queries[:, 1]]
    hc = np.bincount(train[:, 0], minlength=total)
    gc = np.bincount(train[:, 2], minlength=total)
    rc = np.bincount(train[:, 1], minlength=len(r))
    n = len(e) - (4093 if complex_model else 1876)
    blocks = []
    for lo in range(0, len(queries), 128):
        hi = min(lo+128, len(queries)); qb = q[lo:hi]; b = queries[lo:hi]
        gold = torch.as_tensor(gold_cache[lo:hi], device='cuda')
        tau = (qb[:, None] * e[oldsample[lo:hi]]).sum(-1).std(-1).clamp_min(1e-8)
        z = ((qb[:, None]*pe[candidates[lo:hi]]).sum(-1)-gold[:, None])/tau[:, None]
        dz = (qb[:, None]*delta[candidates[lo:hi]]).sum(-1)/tau[:, None]
        future = z+epoch_steps*dz
        known = np.array([sum(c >= n for c in truths[int(h), int(r)]) for h, r, g in b])
        fixed = np.column_stack([teacher[lo:hi], 11-teacher[lo:hi],
                                 (gold/tau).cpu().numpy(), qb.norm(dim=-1).cpu().numpy(),
                                 np.log1p(hc[b[:, 0]]), np.log1p(rc[b[:, 1]]),
                                 np.log1p(gc[b[:, 2]]), known])
        current = torch.stack([z[:, :32].max(-1).values, z.max(-1).values,
                               z.topk(4, dim=-1).values.mean(-1), z.mean(-1),
                               z.std(-1), (z>0).double().mean(-1), F.softplus(z).mean(-1)], 1).cpu().numpy()
        lookahead = torch.stack([(z+dz).max(-1).values, future.max(-1).values,
                                 (future>0).double().mean(-1), dz.max(-1).values,
                                 dz.abs().mean(-1)], 1).cpu().numpy()
        blocks.append(np.column_stack([fixed, current, lookahead, np.full(len(b), .5)]).astype(np.float32))
    x = np.concatenate(blocks); assert x.shape==(len(queries), 21) and np.isfinite(x).all()
    torch.cuda.synchronize()
    return x, dict(seconds=time.monotonic()-tick, bytes=x.nbytes,
                   sampled_candidates=128, old_scale_samples=64, full_candidate_feature_scan=False)

def proposal(initial, prefix, new, tape, n, complex_model):
    torch.cuda.synchronize(); tick=time.monotonic()
    e=torch.nn.Parameter(prefix['E'].cuda().clone()); r=initial['R'].cuda()
    e.register_hook(lambda g: torch.cat([torch.zeros_like(g[:n]),g[n:]]))
    opt=torch.optim.Adam([e],lr=.007); opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    j=int(np.flatnonzero(tape['epochs']==11)[0])
    loss=cx.learning(e,r,new,tape,j) if complex_model else base.learning(e,r,new,tape,j)
    before=e.detach().clone(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    delta=(e.detach()-before).cpu(); assert torch.count_nonzero(delta[:n])==0
    torch.cuda.synchronize()
    return delta,dict(seconds=time.monotonic()-tick,step=j,host_state_untouched=True)

def evaluate(score, safe, final):
    events = safe & (final>10); methods = {}
    for name, s in score.items():
        safe_ap = float(average_precision_score(final[safe]>10, s[safe])) if events.any() else None
        policies = {}
        for policy, ids in [('all_reference', np.arange(len(s))), ('current_safe', np.flatnonzero(safe))]:
            order = ids[np.argsort(-s[ids], kind='stable')]; budgets = {}
            for fraction in [.05, .1, .2]:
                k = int(np.ceil(len(ids)*fraction)); chosen = order[:k]; hits = int(events[chosen].sum())
                budgets[str(fraction)] = dict(selected=k, hits=hits,
                    recall=hits/int(events.sum()) if events.any() else None,
                    precision=hits/k if k else None, already_dropped_selected=int((~safe[chosen]).sum()))
            policies[policy] = budgets
        methods[name] = dict(safe_ap=safe_ap, policies=policies)
    return dict(n=len(safe), safe_at_prefix=int(safe.sum()), already_dropped=int((~safe).sum()),
                new_events=int(events.sum()), final_drops=int((final>10).sum()),
                rescued=int(((~safe)&(final<=10)).sum()), methods=methods)

def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic(); status('prepare'); (ROOT/'cache').mkdir(exist_ok=True)
    models={name:joblib.load(f'/data/lc/untzh/risk58_20261013/cache/{name}.joblib') for name in ['tree_current','tree_new_event']}
    # Current tree uses 16 features; new-event tree uses actual Adam lookahead too.
    for name, model in models.items():
        cols=model.steps[0][1].transformers_[0][2]
        expected=list(range(15))+[20] if name=='tree_current' else list(range(21))
        assert list(cols)==expected, (name, cols)
    reports={}
    for seed in [43,17,29,'complex17']:
        complex_model=seed=='complex17'
        if complex_model:
            src=Path('/data/lc/untzh/complex_transfer118_20261014'); n,total=24567,28660
            z=base.load('/data/lc/untzh/wn_transfer112_20261014/cache/train.npz')
            alltrain=np.concatenate([z['old'],z['new']]); modes=['standard','low']
        else:
            src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
            n,total=11258,13134; z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
            alltrain=np.concatenate([z['old'],z['new'],z['extra']]); modes=['low']
        truths=core.truthmap(z['old'],alltrain[len(z['old']):])
        tape=base.load(('/data/lc/untzh/wn_transfer112_20261014' if complex_model else str(src))+'/cache/learning.npz')
        epoch_steps=int((tape['epochs']==11).sum()); proposals={}
        cohort=base.load(src/'cache/cohort.npz'); saved=base.load(src/'evaluation/queries.npz')
        queries=dict(old_train=saved['old_train'],old_valid=saved['old_valid'] if complex_model or seed==43 else base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid'])
        for split in ['old_train','old_valid']:
            phasepath=src/'evaluation'/f'low-{split}-phases.npz' if complex_model else Path(f'/data/lc/untzh/fb_phase117_20261014/seed{seed}/low/{split}-phases.npz')
            with np.load(phasepath) as phases: teacher=phases['old_candidates'][:,1]
            ids=cohort['ids'] if split=='old_train' else np.flatnonzero(teacher<=10)
            q=queries[split][ids]; teacher=teacher[ids]
            if complex_model or seed==43: gold=base.load(src/'cache'/f'fast-{split}.npz')['gold'][ids]
            else: gold=base.load(f'/data/lc/untzh/fast_old_eval107_20261014/seed{seed}/{split}.npz')['gold'][ids]
            st=time.monotonic(); candidates,oldsample=sample(q,truths,n,total,1190017+(0 if complex_model else int(seed)))
            sampling_seconds=time.monotonic()-st
            np.savez(ROOT/'cache'/f'{seed}-{split}-sample.npz',ids=ids,queries=q,candidates=candidates,oldsample=oldsample)
            for mode in modes:
                key=f'{seed}-{mode}-{split}'; status('features',case=key)
                path=src/'cache' if complex_model else Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{seed}/cache')
                initial=torch.load(path/('standard-initial.pt' if mode=='standard' else 'initial.pt'),map_location='cpu',weights_only=True)
                prefix=torch.load(path/('standard-prefix10.pt' if mode=='standard' else 'prefix10.pt'),map_location='cpu',weights_only=True)
                if mode not in proposals:
                    proposals[mode]=proposal(initial,prefix,z['new'],tape,n,complex_model)
                    torch.save(proposals[mode][0],ROOT/'cache'/f'{seed}-{mode}-next-adam.pt')
                delta,proposal_cost=proposals[mode]
                x,cost=features(initial,prefix,q,teacher,gold,alltrain,truths,candidates,oldsample,total,complex_model,delta,epoch_steps)
                score={k:v for k,v in core.ex.rule_scores(x).items() if k in ['boundary','pressure32','pressure128','lookahead']}
                score['random']=np.random.default_rng(1190200).random(len(x)); predict_cost={}
                for name,model in models.items():
                    st=time.monotonic(); score[name]=model.predict_proba(x)[:,1]; predict_cost[name]=time.monotonic()-st
                # Cached current ranks are a costly diagnostic/control, not a cheap-feature claim.
                pp=src/'evaluation'/f'{mode}-{split}-phases.npz' if complex_model else phasepath
                with np.load(pp) as phases: current=phases['epoch10'][ids,1]
                score['cached_full_current_rank']=current.astype(float)
                np.savez(ROOT/'cache'/f'{key}-predictions.npz',X=x,current=current,**score)
                # Endpoint ranks used only after all predictions have been saved.
                with np.load(pp) as phases: final=phases['epoch20'][ids,1]
                report=evaluate(score,current<=10,final)
                report.update(features=cost,predict_seconds=predict_cost,sampling_seconds_shared=sampling_seconds,
                              proposal_cost_shared=proposal_cost,
                              fixed_learner_horizon='historical next epoch; evaluated here next ten epochs',
                              full_current_rank_cost_is_historical=True)
                reports[key]=report; base.write(ROOT/'summary-running.json',dict(cases=reports))
                print('RISK119',key,'events',report['new_events'],
                      {k:v['policies']['current_safe']['0.05']['hits'] for k,v in report['methods'].items()},flush=True)
    base.write(ROOT/'summary.json',dict(cases=reports,total_seconds=time.monotonic()-tick,no_training=True,
                no_refit=True,historical_cache_and_learner_cost_not_free=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try: main()
    except Exception as exc: status('failed',error=repr(exc)); raise
