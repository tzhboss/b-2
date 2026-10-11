"""Versioned defensive/certificate fixes; frozen v1 and outputs are preserved.

Same quadratic objective, SLSQP options and score margins. Acceptance now
requires tighter original-unit primal checks and finite KKT diagnostics.
"""
import time
import numpy as np
from scipy.optimize import minimize
from rank_constraints import constraints


def solve_min_norm(A,b,*,ftol=1e-10,maxiter=500,primal_tol=1e-10,kkt_tol=1e-6):
    started=time.monotonic()
    A=np.asarray(A,dtype=float);b=np.asarray(b,dtype=float)
    if A.ndim!=2 or b.shape!=(len(A),) or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("finite matrix and matching vector required")
    dim=A.shape[1]
    def rejected(status):
        return np.zeros(dim),{"certified":False,"guarded_primal_verified":False,
            "optimality_verified":False,"status":status,"solver_success":False,"iterations":0,
            "min_primal_slack":None,"stationarity_inf":None,"dual_min":None,
            "complementarity_inf":None,"primal_roundoff_tol_original_units":primal_tol,
            "guard_margin_original_units":1e-8,"KKT_abs_tol":kkt_tol,
            "elapsed_s":time.monotonic()-started}
    zero=np.all(A==0,axis=1)
    if np.any(zero & (b>0)):
        return rejected("exact_zero_row_positive_bound_infeasible")
    keep=~zero
    if not keep.any():
        return np.zeros(dim),{"certified":True,"guarded_primal_verified":True,
            "optimality_verified":True,"status":"redundant_empty_system",
            "solver_success":True,"iterations":0,
            "min_primal_slack":float((-b).min()) if len(b) else 0.,
            "stationarity_inf":0.,"dual_min":0.,"complementarity_inf":0.,
            "primal_roundoff_tol_original_units":primal_tol,"guard_margin_original_units":1e-8,
            "KKT_abs_tol":kkt_tol,
            "elapsed_s":time.monotonic()-started}
    nonzero=A[keep]
    scale=np.max(np.abs(nonzero),axis=1)
    lengths=scale*np.sqrt(np.sum((nonzero/scale[:,None])**2,axis=1))
    a=nonzero/lengths[:,None];rhs=b[keep]/lengths
    if not np.isfinite(a).all() or not np.isfinite(rhs).all():
        return rejected("tiny_nonzero_row_normalization_numerically_unresolved")
    x0=np.zeros(dim)
    # Original row0 is promotion; do not reinterpret first surviving guard as it.
    if len(A) and not zero[0]:
        row=np.where(np.flatnonzero(keep)==0)[0][0]
        x0=max(float(rhs[row]),0.)*a[row]
    result=minimize(lambda x:.5*float(x@x),x0,jac=lambda x:x,method="SLSQP",
        constraints=[{"type":"ineq","fun":lambda x:a@x-rhs,"jac":lambda x:a}],
        options={"ftol":ftol,"maxiter":maxiter,"disp":False})
    d=result.x;slack=A@d-b
    multipliers=np.asarray(getattr(result,"multipliers",[]),dtype=float)
    primal=bool(np.isfinite(d).all() and np.min(slack)>=-primal_tol)
    complete=len(multipliers)==len(rhs) and np.isfinite(multipliers).all()
    stationarity=float(np.max(np.abs(d-a.T@multipliers))) if complete else float("inf")
    dual_min=float(multipliers.min()) if complete else float("-inf")
    complement=float(np.max(np.abs(multipliers*(a@d-rhs)))) if complete else float("inf")
    optimal=bool(complete and stationarity<=kkt_tol and dual_min>=-kkt_tol and complement<=kkt_tol)
    diag={"certified":bool(result.success and primal and optimal),
        "guarded_primal_verified":primal,"optimality_verified":optimal,
        "status":str(result.message),"solver_success":bool(result.success),"iterations":int(result.nit),
        "min_primal_slack":float(slack.min()) if np.isfinite(slack).all() else None,
        "stationarity_inf":stationarity if complete else None,
        "dual_min":dual_min if complete else None,"complementarity_inf":complement if complete else None,
        "primal_roundoff_tol_original_units":primal_tol,"guard_margin_original_units":1e-8,
        "KKT_abs_tol":kkt_tol,"elapsed_s":time.monotonic()-started}
    return d,diag
