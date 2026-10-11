# Prospective resource scheduling amendment for037

The original037 protocol/config and frozen source hashes are retained.
Their GPU1/2 allocation is amended operationally for pending jobs only,
using `configs/rank-capacity37-resource-addendum-v2.json` after034 released
GPUs3/5/6. No new scientific method, seed, dataset, proposal, scorer, solver,
update order, parameter budget, wall-time budget or038 gate is introduced.
All added devices are verified idle RTX3090 cards; GPU0 and active034 GPU4
are excluded. Concurrency changes timing conditions, so costs must not be
interpreted as an equal-cost dominance result.

Two seed17 wrappers and their producer children continue from their original
PIDs without signals or restarts. A new resource scheduler uses the original
pinned V4 stage/dependency/audit/aggregation code with five explicit operational
substitutions: require an existing-running queue at startup; adopt existing workers/pending state and add free GPUs; preserve
canonical jobs and store handoff metadata; mark ownership after the first
canonical save; and release the obsolete scheduler only after exact adopted
worker exit statuses have been captured. The original V4 file is unchanged.

Before handoff, verify the source pins, expected coordinator/job IDs, process
start times/parents/command lines, supplemental source hash, peer claims and
GPU snapshot. Only the verified owned scheduler783380 receives SIGSTOP; its
workers remain in their separate sessions. While the old parent is stopped,
terminal workers remain zombies and Linux proc stat supplies their actual
wait exit statuses. Missing/reused PID handles are errors, not terminal success.
After both adopted workers finish and their exit statuses are captured, kill
only the stopped obsolete scheduler to prevent duplicate pending dispatch.
Subsequent stages are ordinary children of the new scheduler and retain the
original preflight/evaluation/audit flow. Existing outputs cannot be reused
as presumed successful missing handles.

Bounded CPU controls passed: exit0 and exit7 are captured before reaping;
reused PID identities and missing handles are rejected; the exact pinned
source transformations compile. No existing process was signalled or model
read/updated by these controls. Evidence is
`results/capacity-resource-handoff-controls-v2.json`. V2 additionally executes
the transformed startup and terminal path on an empty synthetic queue, checking
that existing job and preflight states are preserved.

The operation records the before canonical, prospective GPU snapshot,
supplemental source/config hashes and rendered queue source under
`operations/rank-capacity37-resource-handoff-v2/`. Canonical retains all original
frozen hashes and adds `resource_handoff` metadata. If startup fails before
ownership transfer, restore the before canonical and resume the old scheduler.
After ownership transfer, retain any error and inspect actual children instead
of resuming the old queue or creating duplicate training.

This amendment does not itself prove model utility or completed handoff.
Confirm new live worker handles and continuing original producer progress
after launch, then require the original037/038 terminal audits and gates.

V1 failed before ownership transfer because its compilation-only control
missed the original V4 `status == registered` guard. The helper automatically
restored the before canonical and resumed the old scheduler. Original wrapper
and producer PIDs continued without signals/restarts. V1 sources, controls,
operation receipt and retained-error.json remain under their original paths.
V2 adds the explicit running-state guard and executed startup control rather
than overwriting that failed attempt or changing scientific inputs.

The real V2 handoff subsequently passed its live startup/source check68 files;
both original producer PIDs continued. All four pending jobs completed their
own original preflights and started training. The adopted seed17 wrappers
terminated with exact captured exit0 statuses (783946 and783945), then the
new scheduler dispatched their unchanged evaluation/audits. Only after both
captures did it release the stopped obsolete scheduler783380. This transition
is recorded in the V2 operation receipt and resource-handoff-v2.log. The new
live owner is790633; future checks must read canonical and not revive the old
queue. The handoff success does not qualify037/038 method utility.
