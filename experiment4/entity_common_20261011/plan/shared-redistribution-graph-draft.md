# Shared-row graph follow-up: implementation draft, not registered

Purpose: test whether TRAIN-scoped prediction protection plus bounded old
competitor redistribution improves new-answer retrieval while retaining old
TRUE and unaffected predicted membership. This is a combination of known
locality, representation stability and constrained optimization mechanisms.
Neither algorithmic novelty nor graph utility is established by022R2.

Keep the full common task: all3753 new entity rows, seeds17/29/43, same initial
DistMult model and audited NewEntityOnly desired weights, frozen relations,
11258 full candidates and all74428 old/12406 new directed evaluation queries.
No additional data, source weights, TEST-selected scopes or favorable-seed
selection. Incremental time includes source SGD, scope preparation and update.

Implementation gates now include023R3 cache controls and024 TRAIN scope
preparation. Fixed binary-tree guard scores have a separately disclosed
arithmetic path; retain the original common evaluator for final method
comparison. Full original68334x11258 guard scoring costs12.6–14.6s/seed with
6.384GB peak allocated memory in024. Initial full/scoped compiled members are
235882/221202 (17),234610/220056 (29),234090/219292 (43), grouped into about68260
queries. These are initial-model guards, not all possible old knowledge.
Only15684/15502/15505 prediction keys initially retain their complete original
Top10 after random expansion; initial P ineligibility is substantial and must
stay explicit. TRUE initial ineligible rows are350/347/276. This does not
replace the original old-test denominator or erase initial loss.

Required paired arms: frozen old rows with full prediction protection (existing
audited012 control), frozen old rows with affected-query prediction exemption,
and the same exemption with bounded selected old competitor rows. TRUE anchors
remain protected in all arms. Affected keys are the head/relation keys in
newTRAIN; old query damage on these keys remains in the global metrics.
Changing the prediction contract must be disclosed as a distinct arm.

Protection compilation must use current shared head/gold vectors. Each old
TRUE guard records its actual initial full-pool rank eligibility. For
unaffected query keys, compile each retained old predicted Top10 member as a
gold guard. Protecting only an old tenth-score cutoff is insufficient once
old rows change. Explicitly record ineligible initial guards and baseline
loss rather than describing them as preserved. Eligibility, scope, prediction
members and competitor choice depend only on TRAIN and the common initial
model. Unlabelled competitors are not certified false facts.

Current read-only array inventory: seeds17/29/43 have142240/142187/141854 TRUE
guard rows and68334 prediction query keys each. Full predicted membership
would introduce up to683340 member guards before the affected-key exemption.
The13-entity all-pair solver must not be applied by materializing a guard by
11258-candidate by400-parameter Jacobian: this could require tens of terabytes.
Use batched candidate scoring, query grouping, sparse head/gold incidence and
bounded cutting planes, followed by complete actual rank acceptance. Do not
reduce the registered guard scope to the cuts retained by the solver.

An old competitor can also be a gold or query head on other guards. Such
affected queries require whole-pool rescoring; for unchanged head/gold queries,
changed candidate scores can update counts only under an independently tested
exact comparison convention. Near ties require whole-query rechecks. A final
whole-scope rescore and an independent replay remain necessary. A solver
termination flag or linearized fixed-head residual is not a rank certificate.

Proposal-based fixed witnesses can be infeasible because they exclude an
immovable old competitor. A graph implementation needs an explicit deterministic
witness rule that distinguishes adjustable competitors from frozen ones, plus
bounded unchanged fallback. Any alternative witness mechanism needs its own
correctness controls before registration; do not tune witnesses on VALID/TEST.

Before registration, fix the old-row budget and TRAIN selection rule,
representation importance weights, per-row movement radii, update ordering,
cut/iteration budgets and fallback policy. Preflight must cover changed head,
changed gold, candidate aliasing, every committed step's full protected scope,
input immutability and replay. A small preflight is an implementation gate,
not a replacement for all3753 rows and the three complete seeds.

Utility screen should retain the020 directional requirements: overall newMRR
at least98% of the same source, old Top10 loss at most75% of source and each
seed nonworse, old-head/new-gold MRR at least98% and Hits10 at least95%. Joint
must be nonworse than its scoped frozen-old control on old loss and directional
new retrieval. Freeze the final exact screen and resource cap before inspecting
new graph utility. Preserve all negative attempts; do not reopen the closed
fixed Mahalanobis coefficient search.
