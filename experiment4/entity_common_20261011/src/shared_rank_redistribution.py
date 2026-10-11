"""Exact shared DistMult rank guards and bounded local redistribution.

SLSQP uses sequential quadratic subproblems for a bilinear model. Its success
flag is never a rank certificate. Acceptance requires actual full-pool ranks,
the fixed witness inequalities, and registered movement bounds. This small
problem implementation establishes correctness, not graph-scale efficiency.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RankGuard:
    head: int
    relation: int
    gold: int
    max_rank: int
    kind: str = 'true'


def scores(E, R, head, relation):
    return E @ (E[head] * R[relation])


def rank_receipt(E, R, guards):
    ranks = np.array([1 + np.count_nonzero(
        (s := scores(E, R, g.head, g.relation)) > s[g.gold])
        for g in guards], dtype=np.int64)
    limits = np.array([g.max_rank for g in guards], dtype=np.int64)
    return dict(passed=bool((ranks <= limits).all()), ranks=ranks.tolist(),
                violations=np.flatnonzero(ranks > limits).tolist())


class SharedGuardProblem:
    def __init__(self, E, R, mutable_ids, guards, witness_model, margin=0.):
        self.base = np.array(E, dtype=np.float64, copy=True)
        self.R = np.array(R, dtype=np.float64, copy=True)
        self.ids = np.asarray(mutable_ids, dtype=np.int64)
        self.guards = tuple(guards)
        assert len(np.unique(self.ids)) == len(self.ids) and len(self.ids) > 0
        assert self.ids.min() >= 0 and self.ids.max() < len(E)
        assert np.isfinite(self.base).all() and np.isfinite(self.R).all()
        assert witness_model.shape == self.base.shape and margin >= 0
        self.row = {int(e): i for i, e in enumerate(self.ids)}
        self.dim = E.shape[1]
        self.margin = float(margin)
        pairs, witnesses = [], []
        for j, g in enumerate(self.guards):
            assert 0 <= g.head < len(E) and 0 <= g.gold < len(E)
            assert 0 <= g.relation < len(R) and 1 <= g.max_rank <= len(E)
            s = scores(witness_model, self.R, g.head, g.relation)
            candidates = np.delete(np.arange(len(E)), g.gold)
            # Deterministic fixed witness: highest target scores, ties by ID.
            order = np.lexsort((candidates, -s[candidates]))
            allowed = set(candidates[order[:g.max_rank - 1]].tolist())
            witnesses.append(sorted(allowed))
            pairs.extend((j, int(c)) for c in candidates if int(c) not in allowed)
        self.pairs = tuple(pairs)
        self.witnesses = witnesses

    def model(self, x):
        E = self.base.copy()
        E[self.ids] = np.asarray(x).reshape(len(self.ids), self.dim)
        return E

    def values(self, x):
        E = self.model(x)
        return np.array([np.dot(E[g.head] * self.R[g.relation], E[g.gold] - E[c])
                         - self.margin for j, c in self.pairs
                         for g in (self.guards[j],)], dtype=np.float64)

    def jacobian(self, x):
        E = self.model(x)
        J = np.zeros((len(self.pairs), len(self.ids), self.dim))
        for i, (j, c) in enumerate(self.pairs):
            g = self.guards[j]
            relation = self.R[g.relation]
            query = E[g.head] * relation
            # Sum contributions when head==gold or head==candidate.
            if g.head in self.row:
                J[i, self.row[g.head]] += relation * (E[g.gold] - E[c])
            if g.gold in self.row:
                J[i, self.row[g.gold]] += query
            if c in self.row:
                J[i, self.row[c]] -= query
        return J.reshape(len(self.pairs), -1)


def bounded_redistribution(E, R, mutable_ids, target_rows, weights, radii,
                           guards, options):
    """Return shared rows or unchanged base. A fallback has no new-utility claim.

The objective is distance to the proposed new rows plus an explicitly weighted
distance to unchanged old rows. Target rows and parameter scope come from TRAIN
in a future graph run. Witness choice is fixed before the optimization.
    """
    ids = np.asarray(mutable_ids, dtype=np.int64)
    base = np.array(E, dtype=np.float64, copy=True)
    before = base[ids].copy()
    target = np.array(target_rows, dtype=np.float64, copy=True)
    weights = np.asarray(weights, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    assert target.shape == before.shape and np.isfinite(target).all()
    assert weights.shape == radii.shape == (len(ids),)
    assert (weights > 0).all() and (radii >= 0).all()
    assert np.isfinite(weights).all() and np.isfinite(radii).all()
    assert rank_receipt(base, R, guards)['passed'], 'Unsafe starting guard scope'
    witness_model = base.copy(); witness_model[ids] = target
    problem = SharedGuardProblem(base, R, ids, guards, witness_model,
                                 options['guard_margin'])
    def objective(x):
        d = x.reshape(target.shape) - target
        return .5 * float(np.sum(weights[:, None] * d * d))
    def gradient(x):
        return (weights[:, None] * (x.reshape(target.shape) - target)).ravel()
    def ball_values(x):
        d = x.reshape(before.shape) - before
        return radii ** 2 - np.einsum('ij,ij->i', d, d)
    def ball_jac(x):
        d = x.reshape(before.shape) - before
        J = np.zeros((len(ids), len(ids), before.shape[1]))
        for i in range(len(ids)): J[i, i] = -2 * d[i]
        return J.reshape(len(ids), -1)
    constraints = [dict(type='ineq', fun=ball_values, jac=ball_jac)]
    if problem.pairs:
        constraints.append(dict(type='ineq', fun=problem.values, jac=problem.jacobian))
    opt = minimize(objective, before.ravel(), jac=gradient, method='SLSQP',
                   constraints=constraints,
                   options=dict(ftol=options['ftol'], maxiter=options['maxiter']))
    candidate = problem.model(opt.x)
    receipt = rank_receipt(candidate, R, guards)
    movement = np.linalg.norm(candidate[ids] - before, axis=1)
    residual = float(np.min(problem.values(opt.x))) if problem.pairs else 0.
    accepted = bool(opt.success and np.isfinite(candidate).all() and
                    receipt['passed'] and residual >= -options['primal_tolerance'] and
                    (movement <= radii + options['movement_tolerance']).all())
    info = dict(accepted=accepted, optimizer_success=bool(opt.success),
                optimizer_status=int(opt.status), optimizer_message=str(opt.message),
                iterations=int(opt.nit), objective=float(opt.fun),
                candidate_ranks=receipt['ranks'], candidate_violations=receipt['violations'],
                min_witness_residual=residual, movements=movement.tolist(),
                witnesses=problem.witnesses,
                certificate='actual full-pool ranks and bounds only; no global optimum guarantee')
    return (candidate if accepted else base), info
