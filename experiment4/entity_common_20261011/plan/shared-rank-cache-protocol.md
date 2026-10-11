# Query-grouped shared-model rank cache correctness

Synthetic implementation controls, not real-data method utility. No TRAIN,
VALID, TEST or model checkpoints are consumed by this control.

For fixed reference members with the same Top-k rank requirement on a query,
all are retained exactly when the lowest current reference score has fewer
than k competitors strictly above it. This includes ties under the existing
rank=1+count(score>gold) definition; it is not a claim that exactly k candidates
remain in a stable-sort Top-k set when ties exist. Keep every reference member
and recompute its actual shared score after parameter changes.

CUDA candidate scoring uses a fixed double precision reduction per query and
candidate, with disabled multiply/add fusion. Padding/reduction order is fixed
by dimension. It intentionally defines a reproducible guard arithmetic path;
do not silently treat it as bitwise identical to the older GEMM guard path or
to CPU scalar sums. Full common model evaluation still needs the unchanged
evaluation implementation and independent audit. Actual full-pool versus
subset-column and alternative query batching scores must match bitwise on
CUDA. Independent CPU scalar scores must meet the fixed1e-10 tolerance and all
rank counts must match exactly in the controls.

The cache stores the whole query/candidate score matrix, not its huge
Jacobian. Proposals update changed candidate columns. Changed query heads
require full-pool shared-model rescoring. Changed gold members require a new
minimum reference threshold; every query with a changed threshold is fully
recounted. For other queries the score comparisons of changed candidates
update counts exactly. Commit only a safe patch from the current cache
version; rejection leaves all state unchanged. Proposing itself is read-only.
Always independently rescore the entire pool after the control commits and at
the end of any graph run. Cache acceptance is not solver optimality or useful
new-knowledge learning.

Required controls:64 randomized multi-row proposals, simultaneously changed
heads/golds/competitors; independent scalar ranks for every reference member;
grouped minimum-score equivalence; CUDA full/subset/batch exact scoring;
accepted and rejected proposals; immutability before commit and on rejection;
stale-patch rejection; full shared-model rescore after every commit; explicit
strict-score ties and changing weakest-reference identity. Dimension7 and37
entities test masking and padded reductions. Graph dimension200 and11258
candidates require a separate registered scaling/source preflight.
