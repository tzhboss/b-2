# Real shared-row solver integration preflight

Eight real new entity updates per arm, seed17, three arms: full prediction
frozen-old, scoped prediction frozen-old, scoped prediction joint-old. This
is integration correctness/cost gating, not the complete three-seed method
comparison. The latter must still cover all3753 entities and seeds17/29/43,
all common74428 old/12406 new evaluation queries and fixed directional gates.
No VALID or TEST reads in this preflight. Source initial and desired weights
are the exact audited common NewEntityOnly model; this exposed development
source is not a fresh heldout model.

Use the024 compiled full/scoped references unchanged. Joint old budget64 rows
selected entirely from old/newTRAIN and initial scores: among initial rank10
barriers where the desired new-answer score exceeds the reference minimum,
vote for the closest higher old nonpositive competitor, excluding the query
head; choose highest vote counts then lowest oldTRAIN degree/ID. Per new entity
choose one selected old competitor by degree/ID. Unlabelled is not known false.
Old penalty4*(1+degree/median-positive-degree), absolute radius10% initial norm;
new radius twice the maximum initial entity norm, centered at initial rows.
Original old/new weights and relations are shared in every score.

Full execution order is seed+271 random permutation of all3753 new IDs.
Preflight explicitly moves first4 entities with a TRAIN-selected old competitor
to the front, then fills to8 in that same permutation, to exercise joint paths.
This changes only preflight coverage and is not a favorable utility selection.
Cutting-plane working set8 rounds,64 rows/round,200 SLSQP iterations, fixed
margins/tolerances. Every accepted patch must pass all actual compiled guards
and absolute movement balls. Failure returns unchanged rows. Complete GPU
score-cache replay after every preflight update and at terminal.

Independent audit replays all patches, row scope and fixed movement centers,
whole-model state hashes, source/artifact identities and unchanged old rows.
Then CPU independently rescales every terminal reference query across all
11258 candidates using the explicit binary tree; all terminal grouped counts
must match exactly and every member stays rank<=10. Intermediate CPU entire
scopes are not replayed. No new method utility or novelty inferred. Preserve
all failures and do not alter budgets based on development scores.
