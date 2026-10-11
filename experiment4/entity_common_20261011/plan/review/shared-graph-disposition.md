# Complete027/030/032 disposition

030 completed all nine unchanged trained models' evaluation and independent
audits. Original027 evaluation failures are retained: the registered config
omitted `cpu_threads_per_job`. No checkpoint was retrained for this repair.
032 independently reconstructed the complete fixed VALID queries and metrics
for all nine jobs; none is unavailable. All three methods received
`NO_GO_FOR_THIS_FIXED_COMBINATION` under the original registered thresholds.

| Three-seed VALID mean | Source | Full frozen | Scoped frozen | Scoped joint |
|---|---:|---:|---:|---:|
| New MRR | .129986 | .129455 | .130039 | .121864 |
| Old Top10 loss | 2.5750% | 1.9086% | 2.0105% | 4.8109% |
| Old-head/new-gold MRR | .027339 | .024879 | .025492 | .024236 |
| Old-head/new-gold Hits10 | 6.8340% | 5.5788% | 5.9275% | 5.5091% |

Full frozen passed aggregate newMRR and old-loss gates but failed directional
new-answer retention. Scoped frozen failed the old-loss and directional gates.
Scoped joint failed all three and the nonworse-than-scoped-frozen gate.
Close these fixed utility attempts. Do not enlarge budgets/radii or search
penalties/seeds. A new mechanism must retain the same final evaluation and
utility thresholds. These are exposed development results, not heldout/SOTA.

034 remains an implementation-equivalence check on the unchanged methods;
its speed results cannot reverse the negative utility decision. Completed
training, complete numerical audits and method usefulness are distinct.

034 has now completed all9 independently audited3753-step trajectories.
The original-state hashes, complete saved replay arrays, terminal weights and
fixed VALID results match. Nine-task average postprocessing decreased from
3700.87s to2289.31s (38.14%); one timing repetition per seed/device under
concurrent conditions. Source SGD was reused and TEST/CPU qualification was
inherited from the identical030 endpoints, not newly rescored across all TEST.
Report and bound cost evidence are in
`results/KGE-ENTITY-COMMON-FULL-INDEX-EQUIVALENCE-034/`.
This completed engineering result leaves all three032 NO_GO decisions intact.

The TRAIN capacity diagnostic in `rank-capacity-followup-draft.md` now supplies
a specific follow-up candidate: frozen old parameters and explicit, certified
query capacity exemptions, with displaced TRAIN truths disclosed separately.
This changes the preservation contract at those keys and is not confirmed
novelty or already demonstrated utility. Continue only with source identity,
split integrity, reference reconstruction and controlled preflight evidence.

The follow-up037/038 is now complete and all six endpoints/utility audits
pass. Both fixed capacity exemption combinations receive NO_GO: directional
new-answer retrieval remains below the original gate. Exempted old TRAIN
top10 truth losses are109/118 across seeds versus5 in the original QP controls.
Complete report/cost disclosure is in038; close this fixed utility route without
changing old gates or negative results.039 is a separate TRAIN-only endpoint
slot-conflict diagnosis, not a trained intervention or utility success.

039 is complete and audited for all three seeds/four policies. Its terminal
canonical has no active processes. Independent CPU fixtures replayed5,764,224
scores in384 complete query rows; full TRAIN scans were GPU-only. At least
95.4%-100% of original blocked source replacements have a fixed-old capacity
barrier (94.8%-100% after exemptions). New-row reordering cannot remove those
particular constraints. No weights were updated and no utility was established.
See039 report.md and mechanism-disclosure.json. Preserve TRUE-only013, task
metric019/020, fixed64-old-row032 and capacity037/038 negative decisions.

A distinct candidate to assess is direct newTRAIN retrieval optimization inside
the already audited rank-feasible set, instead of nearest-source projection.
It cannot remove immutable capacity barriers; its hypothesis is recovery of
useful scores on remaining queries/directions. It requires actual objective
controls, unchanged full/scoped references, bounds and common utility gates.
No expanded radius, relaxed guard or selected favorable seed is authorized by
this diagnosis. This candidate is not yet a useful method or novelty claim.

040 and040R2 both completed six TRAIN-only eight-coordinate preflights,
with full mining/scalar-loss/state checks and32 terminal full-pool CPU query
fixtures each (2,161,536 cells per experiment; same fixture cohort reused,
not doubled independent coverage). Original active-cut control and attempted
feasible-start repair failures are retained. Root cause is an impossible
positive-margin inequality between an immutable tied competitor and reference.
V2 omits immutable-only solver working rows; actual rank acceptance is unchanged.
Strong active-cut TRAIN descent and exhausted-budget unchanged-fallback controls
now pass. This is numerical qualification, not demonstrated useful retrieval.

041 is separately registered: six full3753-row direct TRAIN-loss sweeps,
original full/scoped guard identities, frozen old parameters, original bounds,
complete common TEST and unchanged directional/old-loss gates. Its current
execution and utility are determined only by remote041 canonical/processes
and final independent audits; registration is not a success result.
