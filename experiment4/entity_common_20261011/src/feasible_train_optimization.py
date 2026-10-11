"""Direct TRAIN loss minimization with unchanged shared-rank acceptance.

Known constrained coordinate training; no novelty or utility claim. All old
rows/relations and all compiled guard identities stay fixed. This can improve
within a feasible set, but cannot remove its immutable-old capacity barriers.
"""
import time
from collections import defaultdict
import numpy as np
import torch
from scipy.special import expit
from shared_rank_cache_v2 import fixed_scores
from shared_rank_solver_v2 import LocalPairProblem, GuardSeparator


def score_polynomial(E, R, entity, triples):
    """s(h,r,t) = a*x + b*x^2 + c, including h=t=mutable aliases."""
    triples=np.asarray(triples,dtype=np.int64).reshape(-1,3)
    h,r,t=triples.T
    a=np.zeros((len(triples),E.shape[1])); b=np.zeros_like(a)
    c=np.sum(E[h]*R[r]*E[t],axis=1)
    head=(h==entity);tail=(t==entity);both=head&tail
    a[head&~tail]=R[r[head&~tail]]*E[t[head&~tail]]
    a[tail&~head]=E[h[tail&~head]]*R[r[tail&~head]]
    b[both]=R[r[both]]
    c[head|tail]=0.
    return a,b,c


class TrainLossProblem(LocalPairProblem):
    def __init__(self,E,R,entity,positives,negatives,center,radius,l2,margin):
        super().__init__(E,R,[entity],E[[entity]],[1.],[center],[radius],margin)
        self.pos=score_polynomial(E,R,entity,positives)
        self.neg=score_polynomial(E,R,entity,negatives)
        self.l2=float(l2)
        assert len(self.pos[2]) and len(self.neg[2]) and self.l2>=0

    def score(self,x,polynomial):
        a,b,c=polynomial
        return a@x+b@(x*x)+c

    def objective(self,x):
        x=np.asarray(x).reshape(-1)
        p=self.score(x,self.pos);n=self.score(x,self.neg)
        return float(np.logaddexp(0.,-p).mean()+np.logaddexp(0.,n).mean()+self.l2*np.mean(x*x))

    def gradient(self,x):
        x=np.asarray(x).reshape(-1)
        p=self.score(x,self.pos);n=self.score(x,self.neg)
        ap,bp,_=self.pos;an,bn,_=self.neg
        return (-(expit(-p)[:,None]*(ap+2*bp*x)).mean(0)
                +(expit(n)[:,None]*(an+2*bn*x)).mean(0)+2*self.l2*x/len(x))


class TrainIndex:
    def __init__(self,old,new):
        self.new=np.asarray(new,dtype=np.int64)
        self.incident=defaultdict(list);self.truth=defaultdict(set)
        for h,r,t in np.r_[old,new]:self.truth[(int(h),int(r))].add(int(t))
        for i,(h,r,t) in enumerate(new):
            if h>=7505:self.incident[int(h)].append(i)
            if t>=7505 and t!=h:self.incident[int(t)].append(i)

    @torch.no_grad()
    def examples(self,cache,entity,negative_count):
        positive=self.new[self.incident[entity]].copy()
        assert len(positive),entity
        keys=np.unique(positive[:,:2],axis=0)
        key_index={tuple(map(int,k)):i for i,k in enumerate(keys)}
        ids=torch.as_tensor(keys[:,0],device=cache.device)
        rels=torch.as_tensor(keys[:,1],device=cache.device)
        scores=fixed_scores(cache.E[ids]*cache.R[rels],cache.E).cpu().numpy()
        chosen={}
        for (h,r),s in zip(keys,scores):
            # Non-TRAIN-labelled candidates are competitors, not certified false.
            s=s.copy();s[list(self.truth[(int(h),int(r))])]=-np.inf
            # Stable complete pool order, ties broken by entity ID.
            chosen[(int(h),int(r))]=np.argsort(-s,kind='stable')[:negative_count]
            assert np.isfinite(s[chosen[(int(h),int(r))]]).all()
        negative=np.asarray([(int(h),int(r),int(c)) for h,r,t in positive
                             for c in chosen[(int(h),int(r))]],dtype=np.int64)
        return positive,negative,keys,scores


def optimize_train_row(cache,index,entity,center,radius,l2,negative_count,options):
    started=time.monotonic();before=cache.E[[entity]].cpu().numpy().copy()
    positive,negative,keys,scores=index.examples(cache,entity,negative_count)
    problem=TrainLossProblem(cache.E.cpu().numpy(),cache.R.cpu().numpy(),entity,
                             positive,negative,center,radius,l2,options['guard_margin'])
    initial_loss=problem.objective(before.ravel())
    # Unconstrained means no rank working set yet; the inherited absolute ball
    # remains enforced. Full rank acceptance is mandatory before any commit.
    opt=problem.solve(before,options)
    if not opt.success or not np.isfinite(opt.x).all():
        fallback=cache.propose([entity],before);assert fallback.passed
        return fallback,dict(accepted=False,status='initial_optimizer_failure',
             optimizer_status=int(opt.status),before_loss=initial_loss,after_loss=initial_loss,
             training_positive_count=len(positive),training_negative_count=len(negative),
             history=[],elapsed_s=time.monotonic()-started),positive,negative
    separator=GuardSeparator(cache,[entity],opt.x.reshape(1,-1),[])
    history=[];last_status='direct_train_optimum'
    for iteration in range(options['max_cut_rounds']+1):
        rows=opt.x.reshape(1,-1);patch=cache.propose([entity],rows)
        move=float(np.linalg.norm(rows[0]-center));loss=problem.objective(opt.x)
        bounds=move<=radius+options['movement_tolerance']
        history.append(dict(round=iteration,cuts=len(problem.pairs),violating_queries=int((patch.counts>=cache.k).sum()),
                            optimizer_status=int(opt.status),loss=loss,bounds_ok=bool(bounds)))
        if patch.passed and bounds and loss<=initial_loss+1e-10:
            return patch,dict(accepted=True,status=last_status,before_loss=initial_loss,after_loss=loss,
                 movements=[move],training_positive_count=len(positive),training_negative_count=len(negative),
                 history=history,elapsed_s=time.monotonic()-started,
                 certificate='Actual complete unchanged rank scope, absolute ball, local fixed-TRAIN loss nonincrease'),positive,negative
        if iteration==options['max_cut_rounds']:break
        old_size=len(problem.pairs);problem.add(separator.separate(patch,options['cut_rows_per_round']))
        if len(problem.pairs)==old_size:last_status='no_new_separating_rows';break
        opt=problem.solve(rows,options);last_status=f'slsqp_{opt.status}'
        if not opt.success or not np.isfinite(opt.x).all():break
        if np.min(problem.values(opt.x)) < -options['primal_tolerance']:break
    fallback=cache.propose([entity],before);assert fallback.passed
    return fallback,dict(accepted=False,status=last_status,before_loss=initial_loss,after_loss=initial_loss,
         training_positive_count=len(positive),training_negative_count=len(negative),history=history,
         elapsed_s=time.monotonic()-started,fallback='Unchanged audited feasible row'),positive,negative
