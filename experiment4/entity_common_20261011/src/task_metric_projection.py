"""TRAIN task-score Mahalanobis projection with original-space certificates.

Known quadratic projection. Whitening is implementation, not a scientific
guarantee on unseen facts or on the nonlinear coupled new-entity task.
"""
import time
import numpy as np
from scipy.linalg import solve_triangular
from safe_proposal import nearest_feasible


def query_metric(directions,dimension):
    q=np.asarray(directions,dtype=np.float64);assert q.shape==(len(q),dimension) and np.isfinite(q).all()
    gram=q.T@q;trace=float(np.trace(gram))
    return np.eye(dimension)+(dimension/trace)*gram if trace>0 else np.eye(dimension)


def verify_task_projection(u,A,b,H,d,certificate,options):
    u=np.asarray(u);A=np.asarray(A);b=np.asarray(b);d=np.asarray(d);H=np.asarray(H)
    assert np.isfinite(d).all() and np.isfinite(H).all() and (b>=0).all()
    assert np.allclose(H,H.T,atol=1e-12,rtol=0)
    if not certificate['certified']:
        assert np.array_equal(d,np.zeros_like(d));return dict(fallback=True)
    if len(b):assert np.max(A@d-b)<=options['primal_tolerance']
    ids=np.asarray(certificate['dual_rows'],dtype=np.int64);dual=np.asarray(certificate['dual'],dtype=np.float64)
    assert len(ids)==len(dual) and len(np.unique(ids))==len(ids) and np.isfinite(dual).all() and (dual>=0).all()
    station=H@(d-u)
    if len(ids):
        assert ids.min()>=0 and ids.max()<len(A)
        station=station+A[ids].T@dual
        complement=float(np.max(np.abs(dual*(A[ids]@d-b[ids]))))
    else:complement=0.
    error=float(np.max(np.abs(station)))
    assert error<=options['kkt_tolerance'] and complement<=options['kkt_tolerance']
    # H-norm is the norm for which zero-containing projection is contractive.
    assert np.sqrt(max(0.,d@H@d))<=np.sqrt(max(0.,u@H@u))+options['norm_tolerance']
    return dict(fallback=False,stationarity_inf=error,complementarity_inf=complement)


def nearest_task_feasible(u,A,b,H,options):
    started=time.monotonic();u=np.asarray(u,dtype=np.float64);A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64);H=np.asarray(H,dtype=np.float64)
    assert H.shape==(len(u),len(u)) and np.allclose(H,H.T,atol=1e-12,rtol=0) and np.isfinite(H).all()
    L=np.linalg.cholesky(H);target=L.T@u
    a=solve_triangular(L,A.T,lower=True,check_finite=True).T
    inner=dict(options);inner['kkt_tolerance']=options['kkt_tolerance']/max(1.,float(np.linalg.norm(L,ord=np.inf)))
    z,c=nearest_feasible(target,a,b,inner)
    def failure(reason):
        return np.zeros_like(u),dict(certified=False,status=reason,dual_rows=[],dual=[],elapsed_s=time.monotonic()-started,metric='task_query',inner_status=c['status'])
    if not c['certified']:return failure('weighted_inner_failure')
    d=solve_triangular(L.T,z,lower=False,check_finite=True)
    rows=np.asarray(c['dual_rows'],dtype=np.int64)
    rawdual=np.asarray(c['dual'])/np.linalg.norm(a[rows],axis=1) if len(rows) else np.empty(0)
    certificate=dict(c,dual=rawdual.tolist(),dual_units='original_constraint_rows',metric='task_query',elapsed_s=time.monotonic()-started)
    try:checks=verify_task_projection(u,A,b,H,d,certificate,options)
    except AssertionError:return failure('weighted_original_space_certificate_failed')
    certificate.update(checks);return d,certificate
