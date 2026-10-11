# Direct TRAIN optimization inside original rank guards: preflight040

Motivation:039 identifies immutable-old barriers. This intervention does not
remove them or reinterpret unlabelled predictions as false. It tests whether
nearest-source Euclidean projection leaves useful TRAIN descent directions
inside the existing feasible set.019's Mahalanobis source-score preservation
was negative; this uses actual local TRAIN loss with current shared vectors,
not a different distance to the same source target. Known constrained training
and hard-negative mining are not novelty claims.

Dataset: unchanged FB_CKGE0->1, old7505/new3753/final11258 entities, dimension200,
474 reciprocal relations. Seeds17/29/43 are exposed development. Start from
audited original full/scoped frozen QP endpoints, not capacity exemptions.
Freeze all old E and R. Retain original024 full/scoped guard identities exactly.
Keep original solver margins, iteration/cut limits, random seed+271 entity
order and movement balls about original initial rows (radius2*max row norm).

Each coordinate uses every newTRAIN directed triple in which that entity is
head or tail. Choose ten highest-scoring tail competitors per included query
from the current complete11258 pool, stable entity-ID ties, excluding only
known old/new TRAIN truths. Non-labelled competitors are not certified false.
Minimize mean softplus(-positive score)+mean softplus(negative score)
+0.0005 mean(row^2). Ten negatives and L2 coefficient inherit the common
DistMult training configuration; no coefficient/radius/order/seed search.
Shared head/tail aliases, including self triples, use exact quadratic score
polynomials. Negative competitors are fixed during one coordinate solve.
This local sampled objective is not the global all-TRAIN objective.

SLSQP begins with the unchanged absolute movement ball. Separate violated
original guards with the existing witness rule, retaining fixed bounds and
cuts. Accept only actual complete-scope rank<=10, movement bounds and local
loss nonincrease. Failed solves return the unchanged audited row. No claim
of global optimum or all-query knowledge safety follows from guard coverage.

040 is a preflight: first eight entities in the frozen order, both policies
and all three seeds (six jobs), GPUs0-5 when independently verified idle,
two CPU threads/job,1200s/job. No VALID/TEST queries or new utility selection.
Controls independently verify384 polynomial scalar scores,192 derivative
components (including aliases), direct-loss scalar equivalence, and a toy
TRAIN improvement under full guards. The graph audit replays all eight patches,
independently reconstructs complete-pool hard negatives and scalar losses,
old/rel/budget/state identities, and32 complete terminal CPU query rows/job.
Complete terminal scope is GPU-rescored; bounded fixtures are not a full CPU
terminal certification. Record runtime/memory/fallbacks and source identities.
If any control fails, retain the failure and repair only demonstrated code
errors, never adjust scientific settings to manufacture utility.

A full follow-up requires prior registration, all3753 entities/one sweep/six
jobs, independent full terminal CPU guard audits and original common evaluation
(74428 old TEST/12406 new TEST, unfiltered strict ranks, original old-pool
Top10 eligibility). Fixed VALID old/new1024 queries, seed124, original gates:
newMRR>=98% source; oldTop10 loss<=75% source and each seed nonworse;
old-head/new-gold MRR>=98% and Hits10>=95%. Report directions, TRAIN guard
membership/truth costs, reused-source+original-QP+preparation+new optimization
cost and paired original-QP controls. Independent unchanged endpoint baselines
are reused with exact identities, not new independent training. No heldout,
SOTA, dataset generalization or successful final method claim from preflight.

Traceability: TRAIN objective vs nearest-source objective is the sole new
mechanism. Full/scoped policies separate existing guard selection. All old
parameter stability and parameter budgets unchanged. Further module ablations,
datasets and novelty validation remain necessary for any paper claim.
