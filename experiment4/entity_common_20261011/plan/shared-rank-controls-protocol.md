# Shared-model redistribution correctness and feasibility controls

Synthetic controls only. No graph TRAIN/VALID/TEST or checkpoints are consumed.
No real-data utility, new algorithm, or full-goal completion is inferred.

Objective: minimize weighted squared distance to desired new rows and unchanged
old rows, subject to movement balls and sufficient rank witness inequalities.
The relation rows are fixed; explicitly selected old and new entity rows are
shared as heads, answers, and competing candidates. Existing frozen-query QP
certificates cannot be reused when these rows change.

For each retained TRUE or unaffected prediction guard, choose the highest
max_rank-1 nongold candidates in the target proposal as a fixed witness. All
other candidates must score below the actual shared gold by the fixed margin.
This is sufficient, not necessary, for the rank guard. Witness infeasibility
can reject a feasible rank edit. No alternate witness search is performed.
SLSQP sequential quadratic subproblems solve the bilinear constraints locally;
there is no global-optimality guarantee. Every accepted candidate must satisfy
all actual full-candidate ranks and registered movement bounds as well as
numerical witness feasibility and solver termination. Failure returns unchanged
shared entity rows. Tolerance-only score certification does not suffice for
the exact strict-score rank convention, especially at ties.

Required checks: 64 independent central-difference Jacobian cases with changed
heads, golds, competitors, and same-row aliasing; independent scalar full-rank
oracle; shared 13-entity two-relation redistribution, frozen-old control,
edited-query prediction-protection control, unaffected full Top10 set,
changed-old-head rescore, changed-gold and changed-head false-certificate
counterexamples, exact near-tie rejection, forced optimizer failure fallback,
immutability of input arrays and rows outside the registered parameter scope.

The toy has an old answer at rank10 and an unlabelled competitor above it.
Redistribution may move that competitor below the answer while a new answer
enters Top10. The affected old prediction set changes explicitly. In a real
graph, an unlabelled old candidate is not a known false fact. Whole registered
TRAIN TRUE and unaffected prediction scopes require actual shared rescoring;
all global common old/new evaluation queries remain in the comparison.

Real graph follow-up must separately register TRAIN-only affected query scope,
old competitor selection and cap, update order, importance penalty, resource
budget, all3753 entities and seeds17/29/43, unchanged full candidate evaluation,
directional new-knowledge gates and total source-training plus update cost.
No graph follow-up is authorized by a synthetic success claim alone, but the
user's ongoing task authorizes preparation and execution once its fixed
protocol and correctness checks are established.
