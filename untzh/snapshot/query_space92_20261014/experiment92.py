import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import collections
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/budget_action90_20261014')
import experiment90 as action
route, core, base = action.route, action.core, action.base
ROOT = Path(__file__).resolve().parent
action.ROOT = route.ROOT = route.prev.ROOT = core.ROOT = base.ROOT = ROOT
N, TOTAL = action.N, action.TOTAL
ARMS = ['post_raw', 'post_null', 'post_normmatched']
route.ARMS = ARMS


def build_space(initial, new):
    tick = time.monotonic(); (ROOT/'cache').mkdir(exist_ok=True)
    keys = [set() for _ in range(TOTAL-N)]
    cross_facts = 0
    for h, r, t in new:
        if h < N <= t: keys[int(t)-N].add((int(h), int(r))); cross_facts += 1
        elif t < N <= h: keys[int(h)-N].add((int(t), int(r))); cross_facts += 1
    old, rel = initial['E'][:N].double(), initial['R'].double()
    packed = collections.defaultdict(list); ranks = []; rows = []; vectors = []
    for i, k in enumerate(keys):
        if not k: ranks.append(0); continue
        k = sorted(k); neighbors, relations = np.array(k, dtype=np.int64).T
        q = old[neighbors]*rel[relations]
        rows.extend([i]*len(q)); vectors.append(q)
        norms = q.norm(dim=-1)
        keep = norms > 0
        if keep.any():
            normalized = q[keep]/norms[keep,None]
            _, s, vh = torch.linalg.svd(normalized, full_matrices=False)
            rank = int((s > 1e-10*s[0]).sum()); basis = vh[:rank].T.contiguous()
        else: rank = 0; basis = torch.empty((200,0), dtype=torch.float64)
        ranks.append(rank)
        if rank:
            bucket = min(200, 2**int(np.ceil(np.log2(rank))))
            pad = torch.zeros((200,bucket), dtype=torch.float64); pad[:,:rank] = basis
            packed[bucket].append((i,pad))
        if (i+1)%256 == 0: core.progress('prepare_space', entities=i+1, total=TOTAL-N)
    groups = [dict(indices=torch.tensor([i for i,q in items]), basis=torch.stack([q for i,q in items])) for items in packed.values()]
    flat = torch.cat(vectors) if vectors else torch.empty((0,200), dtype=torch.float64)
    payload = dict(groups=groups, rows=torch.tensor(rows, dtype=torch.long), vectors=flat, ranks=ranks)
    torch.save(payload, ROOT/'cache/query-space.pt')
    ranks = np.array(ranks)
    report = dict(seconds=time.monotonic()-tick, cross_direction_facts=cross_facts,
                  unique_constraints=len(rows), entities=TOTAL-N, zero_rank=int((ranks==0).sum()),
                  full_rank=int((ranks==200).sum()), mean_rank=float(ranks.mean()),
                  rank_quantiles=np.quantile(ranks,[0,.25,.5,.75,.9,1]).tolist(),
                  basis_bytes=sum(g['basis'].numel()*8 for g in groups), tolerance=1e-10)
    base.write(ROOT/'space.json', report)
    for g in groups: g['indices']=g['indices'].cuda(); g['basis']=g['basis'].cuda()
    payload.update(groups=groups, rows=payload['rows'].cuda(), vectors=flat.cuda())
    return payload, report


@torch.no_grad()
def project(delta, space):
    result = delta.double().clone()
    for g in space['groups']:
        ids, q = g['indices'], g['basis']; v = delta[ids].double().unsqueeze(-1)
        result[ids] = (v-torch.bmm(q,torch.bmm(q.transpose(1,2),v))).squeeze(-1)
    return result


def train(arm, initial, prefix, new, tape, data, space):
    tick = time.monotonic(); E,R,opt = core.newmodel(initial,prefix)
    records = []; proposal_seconds = 0.; projection_seconds = 0.
    for j in range(220,440):
        chosen=data['selections'][j-220]
        loss=base.learning(E,R,new,tape,j); gl=torch.autograd.grad(loss,E)[0]
        p,_=action.penalty(E,data,chosen,'absolute_full'); gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        before=E.detach().clone()
        torch.cuda.synchronize(); start=time.monotonic()
        joint_delta=route.ex.proposal(E,opt,gl+alpha*gp)
        proposal_seconds+=time.monotonic()-start
        # Only host gradients enter persistent optimizer state in all three arms.
        opt.zero_grad(set_to_none=True); E.grad=gl; opt.step()
        host=E.detach().clone(); raw=joint_delta[N:]-(host-before)[N:]
        torch.cuda.synchronize(); start=time.monotonic()
        projected=project(raw,space)
        projection_seconds+=time.monotonic()-start
        raw_norm=float(raw.double().norm()); projected_norm=float(projected.norm())
        assert projected_norm <= raw_norm*(1+1e-8)+1e-12
        if arm=='post_null': correction=projected
        elif arm=='post_normmatched': correction=raw.double()*(projected_norm/raw_norm if raw_norm else 0.)
        else: correction=raw.double()
        with torch.no_grad(): E[N:].add_(correction.to(E.dtype))
        actual=(E.detach()[N:]-host[N:]).double()
        changes=(space['vectors']*actual[space['rows']]).sum(-1)
        max_change=float(changes.abs().max()) if len(changes) else 0.
        assert torch.isfinite(E).all() and np.isfinite([raw_norm,projected_norm,max_change]).all()
        records.append(dict(step=j,epoch=int(tape['epochs'][j]),raw_protection_norm=raw_norm,
                            projected_norm=projected_norm,actual_protection_norm=float(actual.norm()),
                            host_step_norm=float((host-before)[N:].double().norm()),
                            cross_train_max_score_change=max_change,
                            cross_train_mean_abs_score_change=float(changes.abs().mean()) if len(changes) else 0.))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:
            core.progress('train',arm=arm,epoch=int(tape['epochs'][j]),max_score_change=max_change)
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/arm; out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records)
    result=dict(seconds=time.monotonic()-tick,steps=220,proposal_seconds_included=proposal_seconds,
                projection_seconds_included=projection_seconds,
                mean_raw_protection_norm=float(np.mean([x['raw_protection_norm'] for x in records])),
                mean_actual_protection_norm=float(np.mean([x['actual_protection_norm'] for x in records])),
                max_cross_train_score_change=max(x['cross_train_max_score_change'] for x in records),frozen_equal=True)
    base.write(out/'training.json',result); return result


def main():
    torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare()
    extra=action.extra_data(truths,data);space,space_cost=build_space(initial,new)
    costs=dict(preparation_seconds=ps+extra,space=space_cost,reused_source=str(route.SOURCE),training={})
    for arm in ARMS:
        costs['training'][arm]=train(arm,initial,prefix,new,tape,data,space)
        core.dump('costs-running.json',costs)
    metrics,es=route.evaluate(truths)
    reference=json.loads(Path('/data/lc/untzh/budget_action90_20261014/summary.json').read_text())['metrics']['absolute_full_uniform']
    metrics['joint_fullabsolute']=copy.deepcopy(reference)
    costs.update(evaluation_seconds=es,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
