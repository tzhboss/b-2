"""Same local solver with reusable immutable reference indexes.

No active experiment imports this module. Dynamic model scores and acceptance
still use the unchanged cache. Static scope identity is checked on every call.
"""
import time
from types import MappingProxyType
from weakref import WeakKeyDictionary
import numpy as np
import torch
from shared_rank_solver_v2 import LocalPairProblem, GuardSeparator as OriginalSeparator


class StaticGuardIndex:
    def __init__(self,cache):
        self.keys_tensor=cache.keys.clone();self.refs_tensor=cache.references.clone()
        groups={}
        for q,g in cache.references.cpu().numpy():groups.setdefault(int(q),[]).append(int(g))
        self.refs=MappingProxyType({q:tuple(gs) for q,gs in groups.items()})
        self.keys=cache.keys.cpu().numpy().copy();self.keys.setflags(write=False)

    def validate(self,cache):
        assert torch.equal(cache.keys,self.keys_tensor),'Static query scope changed'
        assert torch.equal(cache.references,self.refs_tensor),'Static reference scope changed'


_indexes=WeakKeyDictionary()


def static_index(cache):
    if cache not in _indexes:_indexes[cache]=StaticGuardIndex(cache)
    index=_indexes[cache];index.validate(cache);return index


class GuardSeparator(OriginalSeparator):
    def __init__(self,cache,ids,target,preferred_old):
        self.cache=cache;self.ids=np.asarray(ids);self.target=np.asarray(target);self.old=set(map(int,preferred_old))
        assert self.old.issubset(set(map(int,self.ids)))
        index=static_index(cache);self.refs=index.refs;self.keys=index.keys;self.witnesses={}
        self.target_patch=cache.propose(ids,target)


def solve_shared_update(cache,ids,target,weights,centers,radii,preferred_old,options):
    """V2 operations, with static grouping and an identical target-patch reuse."""
    started=time.monotonic();ids=np.asarray(ids,dtype=np.int64);before=cache.E[torch.as_tensor(ids,device=cache.device)].cpu().numpy().copy()
    problem=LocalPairProblem(cache.E.cpu().numpy(),cache.R.cpu().numpy(),ids,target,weights,centers,radii,options['guard_margin'])
    separator=GuardSeparator(cache,ids,target,preferred_old);rows=np.asarray(target,dtype=np.float64).copy()
    delta=rows-problem.centers;norm=np.linalg.norm(delta,axis=1)
    scale=np.minimum(1.,np.divide(problem.radii,norm,out=np.ones_like(norm),where=norm>0));rows=problem.centers+scale[:,None]*delta
    history=[];last_status='direct_target'
    for round_index in range(options['max_cut_rounds']+1):
        patch=separator.target_patch if round_index==0 and np.array_equal(rows,separator.target) else cache.propose(ids,rows)
        movement=np.linalg.norm(rows-problem.centers,axis=1)
        bounds_ok=bool((movement<=problem.radii+options['movement_tolerance']).all())
        history.append(dict(round=round_index,violating_queries=int((patch.counts>=cache.k).sum()),cuts=len(problem.pairs),bounds_ok=bounds_ok))
        if patch.passed and bounds_ok:
            return patch,dict(accepted=True,status=last_status,cut_rounds=round_index,cuts=len(problem.pairs),history=history,
                              movements=movement.tolist(),elapsed_s=time.monotonic()-started,witness_queries=len(separator.witnesses),
                              certificate='Actual all-reference shared ranks and absolute movement balls; no global optimum guarantee')
        if round_index==options['max_cut_rounds']:break
        added=separator.separate(patch,options['cut_rows_per_round']);old_size=len(problem.pairs);problem.add(added)
        if len(problem.pairs)==old_size:last_status='no_new_separating_rows';break
        opt=problem.solve(rows,options);last_status=f'slsqp_{opt.status}'
        residual=float(np.min(problem.values(opt.x))) if problem.pairs else 0.
        if not opt.success or not np.isfinite(opt.x).all() or residual < -options['primal_tolerance']:
            history[-1].update(optimizer_status=int(opt.status),min_pair_residual=residual);break
        rows=opt.x.reshape(before.shape).copy()
    fallback=cache.propose(ids,before);assert fallback.passed
    return fallback,dict(accepted=False,status=last_status,cut_rounds=len(history)-1,cuts=len(problem.pairs),history=history,
                         movements=np.linalg.norm(before-problem.centers,axis=1).tolist(),elapsed_s=time.monotonic()-started,
                         witness_queries=len(separator.witnesses),fallback='Unchanged shared rows; no learning success claimed')
