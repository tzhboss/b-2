# Immutable tied-score working-set repair:040R2

All040 scientific settings, endpoints, first eight entities, original
full/scoped references, margins, balls, TRAIN loss, mining, seeds and audits
remain unchanged. Original040 six jobs are terminal audited preflight only.
Original active-cut control41 initial failure is preserved in
scripts/control_feasible_train41_initial_failed.py. An attempted feasible-start
repair also failed and is retained as
src/feasible_train_optimization_v2_failed_start_repair.py.

Root cause in the2D fixture: a frozen head candidate ties the frozen gold.
It is permitted by the original strict-score rank. The old separator adds
an impossible inequality0>=positive margin for this immutable-only pair.
V2 omits only working-set pairs whose head/gold/candidate are all immutable.
The exact all-reference cache acceptance, strict ranks, positive margin on
mutable pairs and absolute balls remain unchanged. No rank-safe row can make
an immutable-only pair satisfy a new positive margin. Omitting that spurious
solver inequality does not waive an original actual-rank guard.

The original scalar/gradient controls pass. Additional controls force an
actual guard violation, require an active cut and independently check scalar
rank2/local TRAIN descent. With zero cut rounds, an unsafe optimizer result
must return unchanged fallback. No numerical tolerance tuning is allowed.

040R2 reruns the same six TRAIN-only preflights on idle GPUs0-5, then replays
all patches, independently mines complete-pool negatives, computes scalar
TRAIN losses and32 complete terminal CPU rows/job. It must bind the original
040 producer and V2 wrapper/module. No VALID/TEST query or utility gate change.
Six-job pass permits separately registered full training, not a utility claim.
