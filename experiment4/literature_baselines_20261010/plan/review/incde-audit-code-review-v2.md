# IncDE audit v2 and candidate-expansion evaluator: bounded sidecar review

Code review completed; completed-artifact audit and GPU evaluation not performed. Recommend completing the remaining input-binding checks before using a candidate-diagnostic terminal as evidence. Native score, filter and candidate formulas match the frozen protocol. No controller code, canonical, registration, or running job was modified.

## Reviewed versions and evidence

- `code/audit_incde_native.py`: SHA256 `16202de67602dca909ae177ca443dd0c4bfe88403b99a310ec5e2500b9eef0e9`.
- `code/evaluate_candidate_expansion.py`: SHA256 `2442fca171c0752811d4bb3fba38d3985104e23491d276acfbbe3fd071da6f4d`.
- Frozen `configs/protocols/KGE-LIT-005-candidate-diagnostic.json`: SHA256 `e36e51c12d983f5875d326e94f3755d2dccf8b3d39af719a76daacabdd79111a`.
- KGE-LIT-001 master manifest/protocol hashes remain `7d211e0780af166f027623f57668b0ab5e61f160dce6cc875baca228b88a3951` / `c4fe8740f6bf7783fc7893a2004164ec42bfd46d5b06495665a2064809b8a2fb`.
- Source snapshots, fixture code, JSON results and review evidence: `papers/incde-reference/review-v2/`. All line numbers below refer to the hashes above. Versions were unchanged at report creation.

39 focused CPU fixture expectations were satisfied, including deliberate acceptance-gap examples. This count does not mean39 artifact-audit passes. Tests execute exact source fragments with synthetic inputs and do not invoke either full entrypoint. Temporary checkpoint tensors and toy split files are confined to the reviewer-owned directory. No GPU operations or production checkpoint inference were run. CPU fixtures do not establish CUDA argsort/reduction equivalence.

Current read-only binding check: four registered jobs equal their frozen master entries; each copied-source manifest's declared key/digest mapping exactly equals its corresponding master mapping. This checks declarations, not completed live artifact bytes. Four prerequisite `results/KGE-LIT-001-audit/<job>/audit.json` files and candidate output directories were absent at the review snapshot. KGE-LIT-005 protocol was verified directly; its task-root canonical experiment.yaml was absent, so this review makes no canonical registration/status claim.

## Audit v2: fixes verified

- Lines16–23 bind the frozen master/protocol, full job entry, runtime path substitution/cwd/GPU, seed and source commit. Four actual runtime bindings pass these checks; a synthetic changed GPU is rejected for each job.
- Lines43,70–82 reconstruct stable ascending relation-ID order, then all forward queries followed by inverses. Filename/metadata mismatches, float rank arrays, unequal lengths and query permutations are rejected by focused examples. Rank and query count, integer metadata and tested-snapshot bounds are covered.
- Lines55–59 now accept legitimate epoch31 early stopping and complete epoch0–199 logs; a one-epoch log and a nonsequential normal log are rejected. The specific v1 false rejection is repaired.
- Lines83–85 retain independent float64 means separately from native batch100 torch-float32 reciprocal summation. The prior rounding fixture now gives native round4=.5001 and high-precision round4=.5000 as separate values. Hits counting remains exact. CUDA execution/device agreement still requires controller evaluation.
- Lines60–67 inspect selected epoch_id, corresponding source checkpoint byte identity, float64 finite dimension200 embeddings, and entity/relation counts. Synthetic differing source bytes and NaN embeddings are rejected. The limitation that rounded logs cannot certify the unrounded validation-best decision is now accurately disclosed.
- Lines52–53,104–106 restrict aggregate matching to the final report block and reject duplicate rows for a required snapshot. A stale numeric table before the final marker is ignored.
- Independent sampled replay, p2-normalized float64 sigmoid-TransE, historical candidates/filters, target restoration and native double argsort remain correct. Runtime final epoch remains>30 after best state_dict load, so current selected embeddings are correct even if selected checkpoint epoch<=20. Up to960 deterministic post-training query replays per run; no all-score or pre-FWT checkpoint certification.

## Remaining concrete audit gaps

1. **[P2] Copied worktree integrity still trusts its own mutable manifest (lines25–29).** Frozen master binding protects the original repository files, but line26 separately trusts run-local copied-source declarations. An empty manifest executes zero checks; a changed copied loss file paired with its new digest is also accepted by this fragment. Both cases have CPU fixtures. Current four declarations match the master, so there is no observed substitution. Controller follow-up: derive the complete expected `worktree/...` key/digest mapping from the bound master, require exact key/digest equality, and hash actual copied files against it. Continue skipping only the declared regenerated sorted-training file's live-byte comparison; raw splits and executable code stay bound. Producer `source_inputs_unchanged=True` is an assertion, not this independent binding.

2. **[P2] Early-stop logs can omit the required pre-stop epochs (lines55–59).** `normal==range(len(normal))` plus a separately valid stop31 accepts normal epochs0–2 followed by stop31, and even a stop31 line with no normal epochs. Both fragments accept in fixtures. Native source logs every preceding epoch and breaks before logging the stop epoch. Require `normal==list(range(stop_epoch))` for the single early-stop branch, preserve chronological order, and reject any normal epoch at/after that stop. The full200 branch is already stricter. This is a remaining checker acceptance gap, not evidence of truncated actual jobs.

3. **[P3] Final-report header order and complete row schema remain unchecked (lines52–53,104–106).** Headers are substring-tested; swapping Whole_MRR and Whole_Hits@1 still passes when values agree in the synthetic table. The five expected snapshots each need one row, but an extra snapshot row is not rejected. Require exact ordered header tokens, a closing table, and exactly rows0–4. Invalid structural assertions still raise without a structured FAIL report; this is a reporting limitation, not silent PASS.

## Candidate expansion: frozen protocol correspondence

The code directly checks the full frozen KGE-LIT-005 hash at line13 and implements its five fixed arms. It consumes `checkpoint/<dataset>/4model_best.tar`, not a final-epoch file or an earlier snapshot. Direct selected embedding weights remain native-effective final-state embeddings for the reason above.

IDs at lines15–27 agree with author KnowledgeGraph: first-seen train -> valid -> test, h before t, even relation/odd inverse pairs, per-snapshot entity prefixes. Toy five-snapshot split fixtures verify expected IDs, prefix sizes and cumulative filter copies. Native query order is taken from the train4_testt post-training exports; its correctness depends on those exact exports being bound to the completed v2 audit.

Lines28,37 use full selected float64 p2-normalized embeddings and `sigmoid(9 - L1(h+r-tail))`. Historical filters are copied through the tested snapshot, including every train/valid/test truth in both directions. Final filters add later truths. Masking to-10000000 and restoring the queried target are correct. Later-filter counts include truths discovered later about both old and new entity candidates, not just newly introduced entities; this matches the declared full-filter arm.

| Arm | Implemented candidates/filter/rank |
|---|---|
| native_replay | old prefix / old cumulative / double argsort |
| candidate_expansion_only | final prefix / same old cumulative / double argsort |
| strict_rank_old_candidates | old prefix / old cumulative / 1+strictly-greater count |
| strict_rank_final_candidates | final prefix / same old cumulative / 1+strictly-greater count |
| current_filter_final_candidates | final prefix / final cumulative / 1+strictly-greater count |

Exact computation fragments agree with independent scalar strict-count checks on toy queries. Tests verify final strict rank>=old strict rank, current-filter strict rank<=historical-filter final strict rank, removal of a later high-scoring truth only by the full-filter arm, equality/tail restoration, and final snapshot control identities. Equal scores are excluded by strict ranks; ties are counted after historical filtering over final candidates. Arbitrary double-argsort tie ordering can change when candidate count changes, so only the strict pair guarantees monotonicity. The code correctly asserts monotonicity only for that pair. All-query historical native-rank equality at line49 is a strong future gate; if CUDA batch32/stride/tie behavior differs from native batch100, it aborts rather than silently substitutes a different tie rule. That gate has not been exercised on actual outputs.

Metrics at lines51–53 are unrounded float64 means with query-count-weighted overall values, as the diagnostic protocol specifies. They differ intentionally from native per-task round4 / overall round3 arithmetic; the native_replay arm guarantees ranks, not identical printed native MRR. Final-snapshot candidate arms are identical pairwise within their respective tie convention; strict versus argsort may still differ with ties. This should be retained as a controller sanity check.

## Remaining evaluator gap

**[P1] PASS input audit is not bound to the actual consumed files (lines12,19,28,32–33,54).** The only prerequisite is `audit['status']=='PASS'`. It does not compare audit job ID, registration/protocol/audit-code hashes, or audit artifact hashes with current registered-job, selected checkpoint, rank exports and raw splits. Checkpoint/audit hashes are recorded after evaluation, but are not compared to the audit manifest. A focused initial-gate fixture accepts a PASS record with another job ID, obsolete checker and stale checkpoint hash. This is not an observed current substitution; no prerequisite audit files yet existed.

Bounded controller fix before evaluation: validate audit identity and frozen native master/protocol/checker version; require native terminal exit0; hash the current five train4_testt rank exports and 4model_best.tar against the source audit artifact manifest; bind raw splits and mapping-relevant code to frozen copied inputs; validate those NPZ schemas/metadata and exact query order via the source audit contract. Record the consumed input digests, runtime device/software and protocol/implementation hashes. Recheck immutable inputs after evaluation or pin their read-only snapshot. All-query replay alone cannot establish dataset/seed/source provenance or that altered non-sampled ranks came from the audited output.

The source audit's artifact manifest includes the checkpoint and rank files, so this fix can reuse existing digests. Raw split binding needs the master/copied-input contract because those raw-file digests are not included in its output artifact manifest. A same-ID audit built by the repaired checker should explicitly identify its checker hash; do not accept a legacy PASS as equivalent without reviewing its contract.

## Handoff

Code formulas and the repaired v1 regressions are supported by the bounded fixtures. Remaining priorities: bind copied inputs to the master and candidate inputs to the completed source audit; tighten the early-stop sequence check. Reports/fixtures do not certify completed artifacts or establish any measured candidate effect. Controller retains all execution, canonical and promotion authority.
