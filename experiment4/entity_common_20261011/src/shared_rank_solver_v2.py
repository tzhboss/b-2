"""Bounded local shared-row optimization with actual full-scope separation.

Known constrained optimization and rank witnesses, not an algorithmic novelty
claim. A cutting plane list is only a solver working set. Acceptance checks
all references through the exact shared rank cache. No global optimum claim.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from shared_rank_cache_v2 import fixed_scores


class LocalPairProblem:
    def __init__(self,E,R,ids,target,weights,centers,radii,margin):
        self.E=np.asarray(E,dtype=np.float64);self.R=np.asarray(R,dtype=np.float64)
        self.ids=np.asarray(ids,dtype=np.int64);self.row={int(e):i for i,e in enumerate(self.ids)}
        self.target=np.asarray(target,dtype=np.float64);self.centers=np.asarray(centers,dtype=np.float64)
        self.weights=np.asarray(weights,dtype=np.float64);self.radii=np.asarray(radii,dtype=np.float64)
        assert len(self.ids) and len(np.unique(self.ids))==len(self.ids)
        assert self.target.shape==self.centers.shape==(len(self.ids),self.E.shape[1])
        assert self.weights.shape==self.radii.shape==(len(self.ids),)
        assert np.isfinite(self.target).all() and np.isfinite(self.centers).all()
        assert (self.weights>0).all() and np.isfinite(self.weights).all() and (self.radii>=0).all()
        self.margin=float(margin);assert self.margin>=0
        self.pairs=[];self.seen=set()

    def add(self,rows):
        for pair in rows:
            pair=tuple(map(int,pair));assert len(pair)==4 and pair[2]!=pair[3]
            if pair not in self.seen:self.seen.add(pair);self.pairs.append(pair)

    def vector(self,x,entity):
        return x.reshape(self.target.shape)[self.row[entity]] if entity in self.row else self.E[entity]

    def objective(self,x):
        d=x.reshape(self.target.shape)-self.target
        return .5*float(np.sum(self.weights[:,None]*d*d))

    def gradient(self,x):
        return (self.weights[:,None]*(x.reshape(self.target.shape)-self.target)).ravel()

    def values(self,x):
        return np.asarray([np.dot(self.vector(x,h)*self.R[r],self.vector(x,g)-self.vector(x,c))-self.margin
                           for h,r,g,c in self.pairs],dtype=np.float64)

    def jacobian(self,x):
        J=np.zeros((len(self.pairs),len(self.ids),self.E.shape[1]))
        for i,(h,r,g,c) in enumerate(self.pairs):
            head=self.vector(x,h);gold=self.vector(x,g);candidate=self.vector(x,c);rel=self.R[r]
            if h in self.row:J[i,self.row[h]]+=rel*(gold-candidate)
            if g in self.row:J[i,self.row[g]]+=head*rel
            if c in self.row:J[i,self.row[c]]-=head*rel
        return J.reshape(len(self.pairs),-1)

    def ball_values(self,x):
        d=x.reshape(self.target.shape)-self.centers
        return self.radii**2-np.einsum('ij,ij->i',d,d)

    def ball_jacobian(self,x):
        d=x.reshape(self.target.shape)-self.centers;J=np.zeros((len(self.ids),len(self.ids),self.E.shape[1]))
        for i in range(len(self.ids)):J[i,i]=-2*d[i]
        return J.reshape(len(self.ids),-1)

    def solve(self,start,options):
        constraints=[dict(type='ineq',fun=self.ball_values,jac=self.ball_jacobian)]
        if self.pairs:constraints.append(dict(type='ineq',fun=self.values,jac=self.jacobian))
        return minimize(self.objective,np.asarray(start).ravel(),method='SLSQP',jac=self.gradient,
                        constraints=constraints,options=dict(ftol=options['ftol'],maxiter=options['maxiter']))


@torch.no_grad()
def candidate_query(cache,patch,qi):
    """Shared candidate scores for a query in a read-only patch."""
    headmap={int(q):i for i,q in enumerate(patch.head_queries.cpu().tolist())}
    if qi in headmap:return patch.head_scores[headmap[qi]]
    scores=cache.S[qi].clone();scores[patch.ids]=patch.columns[qi]
    return scores


class GuardSeparator:
    def __init__(self,cache,ids,target,preferred_old):
        self.cache=cache;self.ids=np.asarray(ids);self.target=np.asarray(target);self.old=set(map(int,preferred_old))
        assert self.old.issubset(set(map(int,self.ids)))
        refs=cache.references.cpu().numpy();self.refs={}
        for q,g in refs:self.refs.setdefault(int(q),[]).append(int(g))
        self.keys=cache.keys.cpu().numpy();self.witnesses={}
        self.target_patch=cache.propose(ids,target)

    def witness(self,qi):
        if qi in self.witnesses:return self.witnesses[qi]
        scores=candidate_query(self.cache,self.target_patch,qi).cpu().numpy()
        members=set(self.refs[qi]);potential=[e for e in range(len(scores)) if e not in members]
        head=int(self.keys[qi,0]);mutable=set(map(int,self.ids))
        forced=set()
        if head not in mutable and not (members&mutable):
            baseline=self.cache.S[qi].cpu().numpy();cut=min(baseline[g] for g in members)
            # Already higher immutable competitors cannot be moved below the
            # gold. Keep them in permitted rank slots rather than create an
            # infeasible fixed witness when only new rows are modifiable.
            forced={e for e in potential if e not in mutable and baseline[e]>cut}
        # A removable old competitor is deprioritized to create a rank slot.
        # Frozen high candidates remain allowed before mutable old competitors.
        ordered=sorted(potential,key=lambda e:(e in self.old,-scores[e],e))
        # Other reference members can themselves consume rank slots above the
        # weakest reference. Account for the worst case, avoiding silent loss.
        allowance=max(0,self.cache.k-len(members))
        allowed=set(forced)
        for e in ordered:
            if len(allowed)>=max(allowance,len(forced)):break
            allowed.add(e)
        self.witnesses[qi]=allowed
        return allowed

    def separate(self,patch,max_rows):
        violated=torch.nonzero(patch.counts>=self.cache.k,as_tuple=False).flatten().cpu().tolist()
        rows=[]
        for qi in violated:
            s=candidate_query(self.cache,patch,qi).cpu().numpy();members=self.refs[qi]
            gold=min(members,key=lambda e:(s[e],e));allowed=self.witness(qi)
            excluded=[c for c in range(len(s)) if c not in members and c not in allowed and s[c]>=s[gold]]
            for c in sorted(excluded,key=lambda c:(-s[c],c)):
                rows.append((int(self.keys[qi,0]),int(self.keys[qi,1]),int(gold),int(c)))
                if len(rows)>=max_rows:return rows
        return rows


def solve_shared_update(cache,ids,target,weights,centers,radii,preferred_old,options):
    """Optimize selected rows and return safe read-only patch or unchanged rows."""
    started=time.monotonic();ids=np.asarray(ids,dtype=np.int64);before=cache.E[torch.as_tensor(ids,device=cache.device)].cpu().numpy().copy()
    problem=LocalPairProblem(cache.E.cpu().numpy(),cache.R.cpu().numpy(),ids,target,weights,centers,radii,options['guard_margin'])
    separator=GuardSeparator(cache,ids,target,preferred_old);rows=np.asarray(target,dtype=np.float64).copy()
    # Project desired rows into absolute movement balls about fixed centers.
    delta=rows-problem.centers;norm=np.linalg.norm(delta,axis=1)
    scale=np.minimum(1.,np.divide(problem.radii,norm,out=np.ones_like(norm),where=norm>0));rows=problem.centers+scale[:,None]*delta
    history=[];last_status='direct_target'
    for round_index in range(options['max_cut_rounds']+1):
        patch=cache.propose(ids,rows)
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
