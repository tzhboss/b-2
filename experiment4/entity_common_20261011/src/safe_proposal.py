"""Nearest learned-proposal QP with full constraints and sparse dual witness.

Known Euclidean projection, not a novelty claim. Inputs must have nonnegative
slack so the no-update fallback is feasible. Cutting planes are only a speed
optimization: acceptance checks the original full system and original units.
"""
import time
import numpy as np
from scipy.optimize import minimize


def nearest_feasible(proposal,A,b,options):
    started=time.monotonic();u=np.asarray(proposal,dtype=np.float64);A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert u.ndim==1 and A.shape==(len(b),len(u)) and b.ndim==1
    assert np.isfinite(u).all() and np.isfinite(A).all() and np.isfinite(b).all()
    assert (b>=0).all(),'Zero must be exactly feasible for the registered fallback'
    cap=float(np.linalg.norm(u));dim=len(u)
    def fail(reason,iterations=0):
        return np.zeros(dim),dict(certified=False,status=reason,iterations=iterations,dual_rows=[],dual=[],elapsed_s=time.monotonic()-started)
    if not np.isfinite(cap):return fail('nonfinite_proposal_norm')
    if not len(b) or np.max(A@u-b)<=0:
        return u.copy(),dict(certified=True,status='already_feasible',iterations=0,dual_rows=[],dual=[],
                             primal_violation=max(0.,float(np.max(A@u-b))) if len(b) else 0.,stationarity_inf=0.,complementarity_inf=0.,elapsed_s=time.monotonic()-started)
    row_max=np.max(np.abs(A),axis=1);nz=row_max>0;lengths=np.zeros(len(b))
    lengths[nz]=row_max[nz]*np.sqrt(np.sum((A[nz]/row_max[nz,None])**2,axis=1))
    if not np.isfinite(lengths).all():return fail('nonfinite_constraint_norm')
    # Every Euclidean projection of u onto a zero-containing convex set has
    # norm <= norm(u). Strict Cauchy pruning also retains equality/boundary rows.
    keep=nz & (b<=lengths*cap+options['prune_roundoff_margin'])
    ids=np.flatnonzero(keep);a=A[ids]/lengths[ids,None];rhs=b[ids]/lengths[ids]
    if not np.isfinite(a).all() or not np.isfinite(rhs).all():return fail('normalization_unresolved')
    violation=a@u-rhs;working=list(np.argsort(violation)[::-1][:options['cut_rows_per_round']]);working=[i for i in working if violation[i]>0]
    if not working:return fail('nonpositive_normalized_violation_unresolved')
    d=np.zeros(dim);iterations=0
    for round_id in range(options['max_cut_rounds']):
        aa=a[working];bb=rhs[working]
        result=minimize(lambda x:.5*float((x-u)@(x-u)),d,jac=lambda x:x-u,method='SLSQP',
                        constraints=[dict(type='ineq',fun=lambda x:bb-aa@x,jac=lambda x:-aa)],
                        options=dict(ftol=options['ftol'],maxiter=options['maxiter'],disp=False))
        iterations+=int(result.nit);d=np.asarray(result.x,dtype=np.float64)
        if not result.success or not np.isfinite(d).all():return fail('working_qp_failed: '+str(result.message),iterations)
        full=A@d-b;bad=np.flatnonzero(full>options['primal_tolerance'])
        if len(bad):
            bad_local=np.argsort(a@d-rhs)[::-1]
            fresh=[int(i) for i in bad_local if int(i) not in working and a[i]@d-rhs[i]>0][:options['cut_rows_per_round']]
            if not fresh:return fail('full_original_units_primal_unresolved',iterations)
            working.extend(fresh);continue
        dual=np.asarray(getattr(result,'multipliers',[]),dtype=np.float64)
        if len(dual)!=len(working) or not np.isfinite(dual).all() or dual.min() < -options['kkt_tolerance']:
            return fail('missing_or_invalid_dual',iterations)
        dual=np.maximum(dual,0.);stationarity=float(np.max(np.abs(d-u+aa.T@dual)))
        complement=float(np.max(np.abs(dual*(aa@d-bb))))
        if stationarity>options['kkt_tolerance'] or complement>options['kkt_tolerance']:
            return fail('independent_stationarity_or_complementarity_failed',iterations)
        if np.linalg.norm(d)>cap+options['norm_tolerance']:return fail('projection_norm_bound_failed',iterations)
        return d,dict(certified=True,status='full_qp_certificate',iterations=iterations,cut_rounds=round_id+1,full_rows=len(A),
                      pruned_rows=int((~keep).sum()),dual_rows=ids[np.asarray(working)].tolist(),dual=dual.tolist(),
                      primal_violation=max(0.,float(full.max())),stationarity_inf=stationarity,complementarity_inf=complement,
                      elapsed_s=time.monotonic()-started)
    return fail('registered_cut_round_limit',iterations)


def radius_segment(old,delta,radius):
    """Largest alpha in[0,1] keeping old+alpha*delta inside the fixed ball.

Scaling towards the zero update preserves every nonnegative-slack halfspace.
The radius operation is distinct from the QP certificate and is replayed.
"""
    old=np.asarray(old,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
    assert radius>0 and np.isfinite(radius) and np.isfinite(old).all() and np.isfinite(delta).all()
    assert np.linalg.norm(old)<=radius+1e-10
    if np.linalg.norm(old+delta)<=radius:return 1.
    aa=float(delta@delta)
    if aa==0:return 0.
    bb=2*float(old@delta);cc=min(0.,float(old@old)-radius*radius)
    disc=max(0.,bb*bb-4*aa*cc);root=np.sqrt(disc)
    alpha=(-2*cc/(bb+root)) if bb>=0 and bb+root>0 else (-bb+root)/(2*aa)
    alpha=float(np.clip(alpha,0.,1.))
    # Move one representable step inward, not a result-driven radius change.
    return float(np.nextafter(alpha,0.)) if alpha>0 else 0.


def verify_projection(proposal,A,b,delta,certificate,tolerance):
    """Verifier using the full system; does not call the producer solver."""
    proposal=np.asarray(proposal);A=np.asarray(A);b=np.asarray(b);delta=np.asarray(delta)
    assert np.isfinite(delta).all() and (b>=0).all()
    if not certificate['certified']:
        assert np.array_equal(delta,np.zeros_like(delta));return dict(fallback=True)
    if len(b):assert np.max(A@delta-b)<=tolerance['primal_tolerance']
    rows=np.asarray(certificate['dual_rows'],dtype=np.int64);dual=np.asarray(certificate['dual'])
    assert len(rows)==len(dual) and len(np.unique(rows))==len(rows) and (dual>=0).all()
    if len(rows):
        assert rows.min()>=0 and rows.max()<len(A)
        aa=A[rows]/np.linalg.norm(A[rows],axis=1)[:,None];rhs=b[rows]/np.linalg.norm(A[rows],axis=1)
        stationarity=float(np.max(np.abs(delta-proposal+aa.T@dual)))
        complement=float(np.max(np.abs(dual*(aa@delta-rhs))))
    else:stationarity=float(np.max(np.abs(delta-proposal)));complement=0.
    assert stationarity<=tolerance['kkt_tolerance'] and complement<=tolerance['kkt_tolerance']
    assert np.linalg.norm(delta)<=np.linalg.norm(proposal)+tolerance['norm_tolerance']
    return dict(fallback=False,stationarity_inf=stationarity,complementarity_inf=complement)
