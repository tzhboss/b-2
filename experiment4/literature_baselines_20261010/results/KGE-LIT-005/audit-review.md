# KGE-LIT-005 independent CPU candidate-diagnostic audit

Disposition: **PASS for provenance and rank-derived numerical checks, for all four completed jobs.** This is not independent full-score/CUDA replay, paper replication, or automatic result promotion. No native/evaluator/canonical source was edited and no GPU work was launched.

## Implementation and exact evidence

- Review-owned auditor: `code/audit_candidate_diagnostic.py`, SHA256 `433fc9a0a625f54b589990e1bbf2c95a0877c955bc95c2e2ff7b2606839ffb77`. Python standard library + NumPy only; no torch import, model deserialization, scoring or CUDA calls.
- Immutable completed audit report: `plan/review/candidate-diagnostic-audit-initial.json`, SHA256 `08864d9e3a6b4a2e06183ce1f6d1efe87605bcecf41e239e1d9b5628e6cb1c90`.
- Reviewed fixed evaluator SHA256: `313903da5d506a4086bcc0846cb712c764089929f6899f1734aa5b710655af78`.
- Required native source auditor SHA256: `ccb48f4e3da560b26dcdf2d03f6809d46e34f605b539a348a15dc9b778411512`.
- Frozen diagnostic protocol SHA256: `e36e51c12d983f5875d326e94f3755d2dccf8b3d39af719a76daacabdd79111a`.
- Mechanical retry registration: `logs/KGE-LIT-005-mechanical-retry-registration.json`, SHA256 `8f81e97ae44020b11cd35beeba9683bdff0babb342db39a0f8f448fbb72d0b63`.
- Reviewer-owned24 CPU fixture results and selector reproduction: `papers/incde-reference/candidate-diagnostic-audit/cpu-regression-results.json`. Expected good rank controls pass; deliberate corrupted rank/filter/tie/schema cases fail.

## Completed artifact checks

| Dataset | Queries | Checks | Failures | CPU derived audit | Independent score replays |
|---|---:|---:|---:|---|---:|
| ENTITY | 124054 | 2723037 | 0 | PASS | 0 |
| RELATION | 124052 | 2755525 | 0 | PASS | 0 |
| FACT | 124050 | 2742142 | 0 | PASS | 0 |
| HYBRID | 124052 | 2723807 | 0 | PASS | 0 |

Total 496,208 queries and 10,944,511 counted checks, zero failures. Counts include multiple related conditions on each query and do not represent independent statistical observations. The report retains source/output digests and recomputed per-task/overall metrics.

## What was independently checked

Frozen master/protocol/mechanical retry/evaluator/native-auditor hashes; exact master job declaration and runtime path/GPU substitution; native terminal exit0 and bound PASS source audit; exact consumed mapping of five snapshot4 post-train ranks, selected snapshot4 best checkpoint, registered job and native terminal; complete copied input declarations derived from master, and live copied code/raw data bytes. The author-regenerated sorted-training live file remains the declared narrow exception. All bound files and completed output terminal/NPZ bytes are checked again after the audit reads.

All five output NPZ filenames, digests, exact schema, integer one-dimensional arrays, lengths and native cohort identity/order. IDs and cumulative truth sets are independently reconstructed from raw train -> valid -> test splits. Valid/test relation sorting is stable, followed by all forward then inverse queries. Original native ranks/metadata/candidate counts/filter counts are compared to their bound source files. Later-positive counts are independently derived by final minus historical cumulative truth sets.

Every exported rank contributes a reciprocal and Hits1/3/10 indicator to independently recomputed metrics; MRR uses Python math.fsum, not the evaluator metric helper. Per-task metrics, fractions/counts, query-count-weighted overall values and reported query/check totals match completed terminals to absolute1e-12 for real-valued metrics.

Rank checks include candidate bounds, optimistic strict rank <= double-argsort rank, strict final candidates >= strict historical candidates, candidate-count bounds on the strict delta, current filter <= historical-filter final strict rank, filter improvement bounded by later-positive count, tie-position bounds, zero-tie equality, and equality when no later truths exist. Final snapshot requires native=expanded, all three strict controls equal, and zero later truths. Arbitrary double-argsort candidate expansion is not wrongly required to be monotone when ties exist.

## Mechanical selector correction and earlier-review correction

The old prefix selector matched6 final-snapshot rank files: five post_train files plus `ranks/train4_test4_pre_fwt.npz`, then failed assert5 before output creation/inference. This is reproduced against the actual completed RELATION source audit artifact manifest, which includes all19 native rank exports.

The fixed evaluator constructs the exact set `ranks/train4_test0..4_post_train.npz`, selects by membership, and asserts exact set equality. The actual manifest now selects exactly five post-training ranks; a missing required file is rejected. Checkpoint and declaration hashes remain in the consumed mapping. Scoring, filters, five arms and frozen protocol are unchanged by this mechanical repair.

My prior v3 synthetic manifest omitted the legitimate pre-FWT entry, causing the old-selector failure to escape review. The v3 no-blocker conclusion for that old selector was incorrect. This report supersedes that conclusion for selector completeness; the original review, failed evaluator/coordinator records and fixtures remain preserved. New fixtures include the real complete manifest. Existing fixed copied-input and early-stop chronology contracts remain unchanged: normal epochs0..stop-1, chronological normal+single-stop, or full epochs0..199.

Existing native PASS may be reused when its ID/dataset/master/protocol/required-checker identity and consumed hashes still match. This audit verified those bindings for each completed job; it did not rerun or overwrite native audits. Coordinator stop/reuse orchestration and canonical changes remain controller-owned.

## Reusable CLI and output behavior

From the task root:
```bash
/data/conda/envs/test/bin/python code/audit_candidate_diagnostic.py
/data/conda/envs/test/bin/python code/audit_candidate_diagnostic.py --job-id incde-relation-s3407 --output plan/review/candidate-diagnostic-audit-relation-next.json
```

Omitting job IDs checks all four registered jobs. An absent candidate terminal is WAITING; partial outputs are not audited. A malformed completed artifact is FAIL with recorded errors. Exit0 means all requested jobs passed this limited audit; exit1 means at least one failed; exit2 means waiting without failures. JSON is printed; an optional output must be under plan/review, uses exclusive creation and never overwrites an old report. Reusable paths are root-relative. Reports are sidecar evidence, not canonical promotion.

## Limits

No checkpoint embeddings or full model scores are independently recomputed. Equality of exported native ranks to bound native exports checks export consistency, not independent CUDA inference. Tie counts are range/consistency checked and their summaries recomputed, not recovered from independent full scores. Rank identities are necessary conditions and cannot alone authenticate every strict-rank score decision. Source native PASS is bound and consumed, with its sampled replay/unrounded validation-best/single-seed/historical-competitor limitations retained.

No significance/new-algorithm/SOTA or paper-five-run-mean conclusion is drawn from this CPU audit. Controller retains execution, canonical and promotion authority.
