# Bounded local shared-rank working-set solver controls

Synthetic correctness gate; no graph data or utility. Same strict-score rank
cache and binary-tree FP64 acceptance path as023R3. Relation parameters fixed.
The local objective is weighted distance to desired new rows and unchanged
old rows, with absolute movement balls about fixed initial centers. This is
known constrained optimization, not confirmed new algorithmic contribution.

Only violated full-scope query groups produce working-set constraints. For
each such query retain all reference members, select a fixed set of permitted
nonreference competitors based on proposed scores and entity IDs, prioritizing
unchanged competitors over registered adjustable old competitors. The number
of nonreference exceptions is max(0,k-number_of_reference_members), sufficient
but potentially conservative. Selected old candidates may be forced down to
make space for new answers. References are never declared false competitors.
Witnesses can be infeasible and do not prove no rank-feasible edit exists.

Choose the current weakest reference in a violated query, add worst excluded
competitors up to64 per round, and solve the accumulated bilinear inequalities
using SLSQP and analytic shared head/gold/candidate derivatives. After each
solve check every registered reference group through the actual complete
shared model. The working set is never the acceptance scope. Bound8 cut rounds,
500 iterations per solve, fixed margins/tolerances in config. On failure,
budget exhaustion, nonfinite candidate or unsafe actual ranks return unchanged
rows. Optimizer success is not global optimality or learning utility.

Controls:64 independent finite-difference cases for score inequalities,
objective and absolute movement balls, including same-row aliases; constructed
joint versus frozen-old controls; independent scalar all-member ranks; full
cache rescore; analytic toy optimum; solver cannot mutate cache before commit;
forced zero-cut fallback; cumulative radius centers are not reset. The analytic
positive2D toy constraint is x(3-x)/12>=margin, old target x=4 and new target5.
Independent audit must replay saved models and the toy solution before promotion.
