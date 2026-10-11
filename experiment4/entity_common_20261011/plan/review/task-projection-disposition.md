# Fixed task metric: qualified negative decision and next mechanism

Authoritative evidence:019/020 canonicals and audited results, plus021 TRAIN-only scope diagnostic. Original012/013 hypotheses and overall utility gate remain unchanged.019 controls reproduce012 Euclidean model E/R, all old/new rank arrays and original VALID metrics bitwise for all three seeds.

019's six jobs and020's six independent complete VALID replays passed their numerical audits. Both variants fail the new fixed directional utility gate. Euclidean preserves90.9668% of source new-answer VALID MRR and81.6327% Hits10; task-score metric improves these to96.7732% and85.7143%, below98%/95%. Task-score oldVALID loss is81.4264% of source, above the75% limit and worse than Euclidean74.1190%. Exposed TEST: newMRR .13980677→.14077126, oldTop10 loss1.61312%→1.87018%. This is a measurable tradeoff, not a qualified improvement. Close this fixed Mahalanobis candidate; do not vary its coefficient or select favorable seeds.

021 explains a sufficient TRAIN barrier, not all VALID/TEST loss. Among16757 directed newTRAIN facts with old head/new gold,11290/11289/11324 are on initially retained old prediction rank10 boundaries (seeds17/29/43). With oldE/R frozen, the old nine candidates above the boundary remain above it. Registered strict-margin protection keeps each new gold below the boundary, leaving at least ten old candidates above that gold. Independent full old-candidate scoring verified a rank lower bound of11 for every selected fact in both audited Euclidean and task-score endpoints, all three seeds. Roughly67.4% of this TRAIN direction is in that sufficient barrier subset. Scores exactly at the boundary were checked and excluded by actual strict separation; the argument does not assume a tie-free rank definition generally.

TRUE rank10 anchors alone cover8051/7869/7985 of these facts. Therefore exempting only the affected queries from prediction protection would leave substantial barriers under the unchanged frozen-old TRUE contract. This is a limitation of the chosen contract, not an author-paper failure or evidence that protecting real old knowledge is impossible. Keeping an old rank10 answer while adding a new higher-scoring answer can require moving an old competitor below the old answer.

A checked shared-vector2D toy makes that redistribution constructive: lower one old competitor from4 to2.5 in the edited relation, add the new answer at5, preserve the old true answer's rank10, preserve the unaffected relation's full predicted Top10 set, and replay the changed old vector as a query head. The affected old prediction set changes explicitly. This proves only toy feasibility; an old candidate not labelled positive in TRAIN is not certified false knowledge.

Next design draft, not a registered graph method: define affected query keys from newTRAIN only; distinguish known TRUE anchors from prediction membership on unaffected queries; allow a small TRAIN-selected old competitor parameter set under an explicit representation-change penalty. Compare freeze-all, scoped-prediction-only and competitor redistribution controls. Any real joint update must use the actual shared head/tail model, recompute changed query vectors and old gold scores, and verify the entire registered TRUE/unaffected-prediction scope after committing the candidate. Frozen old query features and cutoffs cannot certify a model whose old heads or answers were changed. Record and include affected-query old damage in global common metrics, never hide it by dropping those queries. Preserve full seeds, all3753 new entities and the existing complete common comparison requirement.

Before launching a real method, establish native shared-model correctness and feasibility controls, fixed parameter/data/cost budgets and a new immutable utility screen. No algorithmic novelty, overall heldout victory or whole-goal completion follows from this diagnosis. KGEditor remains in progress according to its live canonicals.

022R2 establishes a bounded shared-row solver and an actual full-pool rank
acceptance check in synthetic controls. The objective weights distance to
unchanged old rows and desired new rows; deterministic proposal-based rank
witnesses produce bilinear constraints solved locally by SLSQP sequential
quadratic subproblems. There is no global optimum or graph-scale efficiency
claim. Derivative checks cover64 changed-head/gold/candidate alias cases
(maximum error2.598636e-9). Independent saved-model scalar replay covers14
guards, complete unaffected Top10 membership, changed-old-head predictions,
parameter scope and movement. The analytic toy solution agrees within8.88e-16:
new rank13→8, old gold rank10 preserved. The affected old prediction set changes.

022 original failed at an incorrect test expectation: a frozen-old target
witness excludes an immovable old competitor and is infeasible, so unchanged
fallback is correct. Its registered source and failure remain unchanged.
022R2 changes that expectation and constructs explicit head/gold stale-feature
counterexamples; the solver and options are unchanged. Head/gold counterexample
fixtures use a separate two-dimensional shared graph where stale cutoffs
approve a candidate but actual old rank changes1→13.

The fixed-witness limitation matters: a target-proposal witness may be infeasible
even when a rank-feasible edit exists with a different witness. Do not treat
such rejection as impossibility of the mechanism. A full-graph implementation
must handle this explicit limitation and bounded fallback, and must avoid
materializing every guard×candidate×400-dimensional Jacobian. Determine actual
TRAIN guard/query counts and full-rescore requirements before registering the
graph run; develop cutting-plane constraints with complete actual-model
acceptance verification. Controls cannot replace all3753 entities and three
seeds or the directional utility screen.

023R3 validates the reusable query-grouped cache. It retains all reference
members and protects their dynamic minimum shared score, exactly equivalent
to all members having strict-score rank<=10 on the same query.64 controls
(24 accepted/40 rejected) and32 independent new cases (3840 member ranks,
16 accepted/16 rejected) pass. Full/subset/query-batch scores match bitwise;
scalar score error is3.55e-15. Shared changed heads require whole-pool rescore;
changed reference minima require whole-query recount. Rejected/stale proposals
cannot change committed state. Original023/023R2 Triton discovery/compiler
failures remain. V2 uses explicit PyTorch adjacent-pair FP64 reduction rather
than the unavailable Triton backend; its arithmetic is disclosed separately
from legacy GEMM.

024 performs a real TRAIN-only scaling/compilation preflight, three seeds,
all68334 query keys and11258 candidates. It passes complete structural member
and affected-key reconstruction plus128 whole-pool CPU score fixtures/seed
(1,441,024 scores/seed), all bitwise exact. Full score cost12.6–14.6s, peak
allocated GPU memory6,383,711,232 bytes; matrix storage6,154,433,376 bytes.
The initial full/scoped references are about235k/220k, grouped into about68k
queries. This removes the dense-Jacobian memory barrier for scoring/acceptance,
not yet for local optimization. Only about15.5k prediction query keys initially
retain all original old-pool Top10 members; this ineligibility is explicitly
reported, along with350/347/276 ineligible selected TRUE rows. There are no
new model updates, VALID/TEST reads or graph method utility results in024.

Next implement bounded cutting-plane local joint optimization with an explicit
adjustable-competitor witness rule. Freeze the parameter selection, degree
importance, update ordering, source-inclusive cost and directional utility
gates before real updates. Final model scores use the unchanged common
evaluator. All3753 new entity updates and all three seeds remain required.
