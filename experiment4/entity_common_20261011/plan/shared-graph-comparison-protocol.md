# Full common new-entity comparison of scoped/shared rank optimization

Hypothesis: limited TRAIN-selected old competitors can reduce the new-answer
retrieval cost of rank protection, relative to scoped frozen-old optimization,
without increasing old damage. This is a known-mechanism combination, no
confirmed algorithmic novelty. Source SGD and scope preparation costs included.

Three paired arms, each seeds17/29/43: same fixed binary-tree guards with full
prediction/frozen-old, scoped prediction/frozen-old, scoped prediction/joint-old.
Keep024 compiled references, initial eligibility and source choices unchanged.
Source is the exact audited NewEntityOnly DistMult initial and desired weights.
Old budget64 TRAIN-selected competitors, one old row per new entity, old penalty
4*(1+degree/median-positive-degree), initial-centered old radius10% norm and new
radius2*maximum-initial-norm. Select old competitors only on newTRAIN rank10
barriers, nearest strictly higher unlabelled nonpositive candidate excluding
head; votes then degree/ID, as frozen in026. Unlabelled is not certified false.
No validation/tail-score selection of old rows or new data/weights.

All3753 new entities are visited exactly once in seed+271 permutation. Relations
and unselected old rows stay unchanged; selected old rows are shared as heads,
golds and candidates. SLSQP working set8 rounds x64 cuts,200 iterations/solve,
fixed margins/tolerances. Candidate acceptance covers all compiled references
and absolute movement balls. On failure return unchanged rows; do not replace
the desired learned vector with an uncertified fallback. Complete guard cache
rescore every128 steps and at terminal. No global optimum claim.

Required prerequisites:025R2 independent synthetic solver audit,024 three-seed
scope/scoring audit,026 successful audited real integration preflight. Full
queue cannot start if026 has retained failures. Per-job preflight protects the
whole scope and exercises8 updates before a full run; this is not the result.
GPU availability and peer claims checked, use3/4/5/6 only while KGEditor uses
its existing claims.7200s per producer; no budget increase after outcome.

Checkpoint is the sole terminal sweep, no new VALID tuning or grid. Use the
same fixed old/new1024 VALID queries for diagnostics and independent full
utility replay; no intermediate utility reads. Evaluate all74428 old TEST and
12406 new TEST directed queries, full11258 candidate pool, unchanged original
GEMM rank=1+count(score>gold) evaluator. Retain old7505-pool Top10 denominator,
initialized same-pool auxiliary, all affected queries, complete new direction
cohorts. Existing datasets have been exposed: this is development comparison,
not new final-holdout confirmation or SOTA proof.

Independent audit covers source/artifact identities, all3753 patched rows and
state hashes, exact update order, fixed movement bounds and selected old scope,
unchanged relations/other old rows, entire terminal query scope on CPU binary
tree and exact terminal counts. Logged intermediate full GPU scopes are not
all independently CPU rescored. Existing common64-row inference fixture audit
and whole-query identity/metric aggregation unchanged. Keep all failures.

Frozen scientific gate on independent VALID replay: mean newMRR>=98% source,
mean oldTop10 loss<=75% source with positive source loss, every seed old loss
nonworse; old-head/new-gold MRR>=98% and Hits10>=95% source. Joint must also be
nonworse than scoped frozen in old loss and new-answer direction MRR/Hits10.
Report all arms/seed variation, costs and failures; no coefficient/seed search
or weakened directional gate after seeing results. Passing numerical audits
does not imply this utility gate passes or the whole goal is complete.
