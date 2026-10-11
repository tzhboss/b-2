# Rank evaluation comparison, 2026-10-10

This is an exposed-cohort supplementary comparison, not a new blind test or a claim of novel optimization. Preserve all preceding results and failures. Primary execution and artifacts reside on the user's experiment4 server.

## Task and fixed inputs

Reuse the two released 256-dimensional DistMult/ComplEx bundles and precisely the same 400 VALID requests per model from audited KGE-RANK-034. Each request starts from the original bundle, updates one candidate tail column, and freezes head queries. Use all bundle entities as candidates, no truth filtering, rank = 1 + count(scores strictly greater than target). Top-k=10; margin=1e-6. TRAIN protection is the original relation's pre-edit Top10 true rows, in original order, capped at5000. External damage uses VALID facts absent from TRAIN under the same relation/cap. No new TEST score computation. VALID and historical TEST exposure preclude heldout claims.

## Methods and information

Reuse audited Direct, Rank fraction.125, TRAIN Support, TRUE+Prediction QP with Rank fallback from034; TRUE-only QP from036; and BOTH declared all-split Support conventions from037. These saved vectors are not refitted. All-split Support reads TRAIN/VALID/TEST labels and is a retrospective information-rich comparison, not a fair TRAIN-only deployment baseline. The target-excluded convention is declared reconstruction, not certified exact author implementation.

Add StrictNull fraction1 and Ridge at prior frozen validation selections (DistMult10, ComplEx100); no parameter search, sample reselection, training or QP reoptimization. Every400 attempt counts, including no-ops/fallbacks. Ridge is a supplementary released baseline, not necessarily an original main-table method.

## Checks and metrics

Before computing results bind bundles, frozen selection, old vector/case records and original audits by SHA256. Verify their hashes occur in audited evidence. Replay ALL saved vector target and locality metrics on the remote runtime. Report target success, no-damage, SafeSucc, mean TRAIN loss, mean external loss, update norm, and descriptive paired SafeSucc differences against Rank. Cases are correlated, and the two query seeds do not represent independent training runs; no significance/SOTA claim.

Independently audit new updates with separate SciPy SVD/projection and a Ridge least-squares objective, independent scalar/vector strict-rank changes for all protected rows, plus direct full candidate rescoring for deterministic samples. Audit source hashes and synthetic tie/Top10 boundary controls. A failure is retained and is not promoted. The old QP certificates are inherited, not newly solved or re-certified.

Two CPU jobs run concurrently with two BLAS threads each; per-model1200s processing budget, excluding transfer and source validation. Efficiency records remote replay latency only; it cannot compare historical training or editing latency. Capture runtime library and host-resource facts. Avoid unrelated processes. Canonical experiment.yaml is sole status source; promote lightweight summaries only after audit.

## Compatibility of literature methods

IncDE/FastKGE/STARK/LKGE can share this raw strict-rank and old-Top10 retention definition when rescored on their own checkpoints, with historical/current candidate scopes separately specified. Their multi-snapshot new-fact training, TransE-like scoring, native dimensions and information sets do not match this single-column edit. Their original MRR or filtered-drop rates must not enter the primary SafeSucc table. A transplanted optimizer is an adaptation requiring a separately fixed common training task, not an original-method reproduction. KGEditor's LM-based model can likewise expose raw entity ranks but our linear-tail QP is not a drop-in editor for it. SAGE retains integrity FAIL. No new continual/LM training is included in this bounded comparison.

Shared-head/tail global protection is a separate contract: retain the audited041 negative results, and do not reinterpret primary frozen-column gains as global safety.
