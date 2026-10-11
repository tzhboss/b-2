# IncDE audit and candidate evaluator v3: final bounded code review

**Disposition: PASS_BOUNDED_CODE_REVIEW — no remaining blocking finding within the reviewed scope.** The v2 integrity and parsing gaps are closed in the supplied versions. This conclusion concerns checker/evaluator code; it is not a PASS for training artifacts, CUDA replay, or diagnostic results.

## Reviewed versions

- `code/audit_incde_native.py`: `ccb48f4e3da560b26dcdf2d03f6809d46e34f605b539a348a15dc9b778411512`.
- `code/evaluate_candidate_expansion.py`: `5cf6ab951bffeaba9fe7dd827018ce30d4d39950d9b1f2330e0207fd6ebc15c4`.
- Frozen KGE-LIT-005 protocol: `e36e51c12d983f5875d326e94f3755d2dccf8b3d39af719a76daacabdd79111a`.

`experiments/KGE-LIT-005/experiment.yaml` exists and its protocol, implementation and required-auditor hashes match these versions. Canonical ownership remains with the controller; this report does not maintain or change its runtime status.

## Verification

46 focused CPU fixture expectations were satisfied, including acceptance of valid native records and rejection of the formerly accepted malformed or changed-input records. Exact source fragments from preserved snapshots were executed with synthetic inputs. Neither production entrypoint was invoked. Reviewed snapshots, fixture code/results and evidence digests are under `papers/incde-reference/review-v3/`.

Six AST comparisons confirm that ID/filter construction, five-arm score/rank logic, selected-checkpoint schema, sampled native replay and native float32 metric arithmetic are unchanged from the already reviewed v2 core. Four current run-local job and copied-manifest declarations independently match the frozen master. This declaration check does not constitute a completed live-artifact audit.

## Closed findings

| V2 finding | Final code and regression evidence |
|---|---|
| Copied manifest trusted itself | Auditor26–31 derives exact keys/digests from bound master and requires declaration equality, then checks live bytes. Empty manifest, redeclared changed loss digest and changed live loss fail. Only author-regenerated sorted-training live bytes are exempt; their manifest declaration must still match. |
| Truncated early-stop chronology accepted | Auditor59–63 requires normal epochs0..stop-1 and chronological epochs exactly normal+single-stop. Legal stop31 and full200 pass. Truncated normal logs, stop-only, stop-before-normal, logging the stop epoch normally, and duplicate stops fail. |
| Header order / complete rows unchecked | Auditor54–57 requires exact six ordered header tokens, exactly five six-column rows0–4 in order. Swapped header, extra/duplicate/reordered/missing rows fail. Stale tables before the final marker are ignored. |
| Evaluator accepted an unbound PASS | Evaluator12–25 binds frozen master/full job, audit ID/dataset/master/protocol/fixed auditor version, terminal exit0, consumed rank/checkpoint/declaration bytes and master-derived copied code/data. Wrong identities/digests, legacy checker, changed job, checkpoint, rank, raw split and nonzero terminal fail. Evaluator67–69 rechecks consumed and raw copied bytes before successful terminal output; post-evaluation checkpoint/code mutations fail. |
| Evaluator structural metadata only inherited | Evaluator45–46 asserts trained/tested snapshot, seed, integer query matrix, integer rank vector and equal lengths. Wrong snapshot/seed, float IDs/ranks and length mismatch fail. Full bounds/order/filter/count contracts are inherited from the required fixed source audit and bound identical bytes. |

The consumed selected `4model_best.tar` and five final-model historical rank files are bound to the required source auditor's artifact manifest. That auditor emits these entries and verifies native schema/contracts. Raw copied mapping/score code and splits are bound to the frozen master before and after evaluation. The evaluator checks the frozen diagnostic protocol directly and records consumed hashes, copied-manifest hash, source-audit hash, implementation/protocol hashes and runtime versions/device.

## Final protocol assessment

Final selected checkpoint is snapshot4's validation-selected best, not the last epoch. Runtime epoch remains>30 after native best-state reload; direct selected current embedding weights therefore match native effective final embeddings even if checkpoint epoch_id<=20. Epoch/source-byte identity and finite float64 dimension200 counts remain checked.

First-seen train -> valid -> test IDs, h before t, even/odd relation inverses, stable relation-sorted forward-then-inverse query order, cumulative tested-snapshot prefixes and filters agree with author source. Final filters accumulate all five snapshots' train/valid/test truths.

The five arms retain normalized float64 sigmoid-TransE scores, historical/final candidates, target restoration and declared double-argsort versus strict-greater ranks. Synthetic native replay, strict monotonicity, later-positive removal, equal-score counts and final-snapshot control identities pass. Later positives may concern old as well as newly introduced entities. Only strict ranks guarantee monotonicity under candidate expansion; double-argsort can reorder ties. All-query native-rank equality remains a required future CUDA gate.

Native batch100 float32 MRR and independent float64 means remain separate. Diagnostic unrounded query-count-weighted means intentionally differ from native task-round4 / aggregate-round3 printing; identical ranks do not imply identical printed metrics.

## Retained limits

Source-audit checkpoint replay samples up to64 queries per post-training file, up to960 per run, and does not certify every score or pre-FWT checkpoint replay. Candidate evaluation checks all final-model historical ranks when actually run. Unrounded validation-best decisions cannot be fully certified from rounded logs; selected epoch/source bytes are checked and this limit is explicit. CPU regressions do not establish actual CUDA reduction/argsort equality or artifact validity. Structural assertions may abort without a structured FAIL JSON; they cannot produce the successful terminal path. Controller orchestration must retain partial/failure artifacts.

No fixed code, canonical, registration or running worktree was edited. No GPU job or production inference was launched. The controller may proceed with registered audits and the fixed evaluator when audited prerequisites exist. No artifact PASS or result promotion is supplied by this report.
