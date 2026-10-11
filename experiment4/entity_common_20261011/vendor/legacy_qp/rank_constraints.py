"""Exact rank constraints for a single changed candidate column, not shared rows."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize


def constraints(query, gap, guard_queries, old_ranks, true_scores,
                competitor_scores, guard_tails, edited_tail, self_thresholds,
                *, top_k=10, margin=1e-8):
    """Return A,b with A delta >= b; only TRAIN guard information is used."""
    other = np.asarray(guard_tails) != edited_tail
    at_boundary = other & (old_ranks == top_k) & (competitor_scores <= true_scores)
    self_rows = ~other
    aa = [np.asarray(query, dtype=float).reshape(1, -1)]
    bb = [np.array([gap], dtype=float)]
    if at_boundary.any():
        aa.append(-guard_queries[at_boundary])
        bb.append(competitor_scores[at_boundary] - true_scores[at_boundary] + margin)
    if self_rows.any():
        aa.append(guard_queries[self_rows])
        bb.append(self_thresholds[self_rows] - true_scores[self_rows] + margin)
    return np.vstack(aa), np.concatenate(bb), {
        "promotion": 1, "boundary_other": int(at_boundary.sum()),
        "edited_true_tail": int(self_rows.sum())}


def solve_min_norm(A, b, *, ftol=1e-10, maxiter=500, tol=1e-7):
    """Fixed SLSQP QP; expose feasibility, KKT diagnostics and solver failures."""
    start = time.monotonic()
    lengths = np.linalg.norm(A, axis=1)
    if np.any((lengths < 1e-14) & (b > tol)):
        return np.zeros(A.shape[1]), {"certified": False, "status": "zero_row_infeasible"}
    keep = lengths >= 1e-14
    a, rhs = A[keep] / lengths[keep, None], b[keep] / lengths[keep]
    q = a[0]
    x0 = max(float(rhs[0]), 0.) * q
    res = minimize(lambda x: .5 * float(x @ x), x0,
                   jac=lambda x: x, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda x: a @ x - rhs,
                                 "jac": lambda x: a}],
                   options={"ftol": ftol, "maxiter": maxiter, "disp": False})
    delta = res.x
    slack = A @ delta - b
    multipliers = np.asarray(getattr(res, "multipliers", []))
    diag = {"certified": bool(res.success and np.isfinite(delta).all()
                              and np.min(slack) >= -tol),
            "status": str(res.message), "solver_success": bool(res.success),
            "iterations": int(res.nit), "min_primal_slack": float(np.min(slack)),
            "elapsed_s": time.monotonic() - start}
    if len(multipliers) == len(rhs):
        diag["stationarity_inf"] = float(np.max(np.abs(delta - a.T @ multipliers)))
        diag["dual_min"] = float(np.min(multipliers))
        diag["complementarity_inf"] = float(np.max(np.abs(multipliers * (a @ delta-rhs))))
    return delta, diag
