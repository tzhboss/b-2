# KGEditor independent CPU auditor review

Final source SHA256: **4153692fac983667dd8a4ae90d034f60a038dc3bdaf93525e7a98d60393914a5**

Owned artifacts only: `code/audit_kgeditor_native.py` and this review. Frozen adapters/config, copied job sources, environments, model/data downloads, canonical records and native output files were not modified. No model inference, training or GPU operation was launched. Script emits one JSON report on stdout; it has no output-writing option and disables Python bytecode writes.

## Run command and result contract

Run after main's producer records terminal completion:

```bash
BASE=/data/lc/tzh/experiment4/literature_baselines_20261010
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$BASE/environment/kgeditor38/bin/python" "$BASE/code/audit_kgeditor_native.py" --root "$BASE"
```

For one registered job:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$BASE/environment/kgeditor38/bin/python" "$BASE/code/audit_kgeditor_native.py" --root "$BASE" --runs kgeditor-e-fb15k237-s0
```

The installed environment supplies PyYAML, NumPy, Transformers/BertTokenizer, PyTorch for **CPU checkpoint metadata loading only**, and TensorBoard's read-only event parser. No author module/model import or forward is performed. Full weight/checkpoint hashing and WordPiece reconstruction may take CPU time and checkpoint loading requires CPU memory. Main may redirect stdout to its own audit artifact if wanted; the script itself never saves files or promotes results.

Exit codes:
- **0 / PASS_WITH_LIMITATIONS**: every required terminal provenance, token/source contract, checkpoint/training/selection evidence and exported arithmetic check passed.
- **1 / FAIL**: a consistency/integrity check failed, or a malformed artifact/dependency prevents a required section from completing. The report lists the failing checks (first40 details, full failure count).
- **2 / INCOMPLETE**: required evidence is missing or canonical still records an active selected job. This is not a successful reproduction.

No terminal.json: validate existing immutable registration/source evidence, list missing final files and defer inspection of potentially partial exports/checkpoints. This avoids diagnosing an actively written JSONL as a final failure. A terminal failure is a failed check even if other files happen to exist. No automatic waiting or restarting.

## Primary binding and canonical snapshot

Master manifest is pinned to:
`configs/experiments/KGE-LIT-004-jobs.json`, SHA256
`314b14fa671d140b579ba5e728afd6efb36349e34404d2f04a8ef33d2442a58b`.

The auditor reads current `experiments/KGE-LIT-004/experiment.yaml` once and reports its exact byte SHA256, checks REGISTERED/id, manifest path/hash and job states. It also reports whether canonical changed during the audit and its ending SHA; runtime canonical is mutable, so it is not hardcoded to one obsolete byte hash. Canonical is never rewritten. A running canonical snapshot cannot support a PASS for an active job even if final-looking files appeared meanwhile; rerun after terminal.

It requires each run's registered-job.json to equal the corresponding master job exactly, then binds current original registered inputs and the **exact registered copied-source subset**. Scope: 14 registered source/config/hash-input files, not the entire copied repository or sidecar provenance. Every corresponding copied file is checked against its registered SHA. All registered raw HF inputs and current pytorch_model.bin files are hashed and checked as registered. The runtime command/cwd/GPU record must equal the scheduler's deterministic source-to-worktree path substitution.

Pinned source commit `3035e7f898b0eaa805542a120aba080f04fd5ba4`; model revision `a103204def5465ac090935c925451eef6692f2cc`; dataset revision `18a560856826fdcfab5b328829f550ce0f9e5a7a`; seed0. Model/dataset author linkage is documented in plan/kgeditor-native.md; this audit checks revision/byte bindings rather than repeating remote retrieval.

## Completion and selected-checkpoint audit

Requires terminal exit0, producer unchanged-input assertion and terminal log SHA. Requires training-complete, exposed metrics and exported JSONL.

- Frozen config: max_epochs20/lr5e-4/edit2/stable8/workers4/KL/seed0/one GPU, with FB extension64 at layers10,11; WN extension16 at layer11.
- Exact full-training optimizer-step counts: **30860 FB** (1543 batches ×20), **14900 WN** (745 ×20). Current epoch19 or20 accommodates the pinned Lightning end-of-fit indexing, but cannot alone establish completion; exact step count is mandatory.
- Selected checkpoint path must stay inside the registered run logger; summary/complete/callback identities and checkpoint SHA agree.
- CPU Lightning checkpoint metadata contains base/editor state keys and all effective hyperparameters; selected epoch0..19 and step count are consistent with epoch-end saving, allowing only the pinned save-before/after-step convention.
- Callback best score/path agrees with training-complete metadata.
- TensorBoard scalar history must include epochs0..19, Eval/hits3, and the maximum logged validation monitor must match the reported selected best score.
- Final exposed evaluation may have a slightly different sampled locality panel and different tie backend; its edit aggregate is not forced to equal the validation monitor numerically. These are separate observations.
- An earlier best checkpoint is acceptable when twenty-epoch training completes. Selected epoch is not incorrectly required to be19.

The metadata branch was exercised using in-memory fixtures; actual terminal checkpoint/event checks await completion of these runs. No false runtime-completion claim is made from a fixture.

## Independent token/source reconstruction

Checks staged raw-input hashes as the exact registered subset; train←edit_train and dev/test←edit_test byte aliases; preflight row retention and omitted rows. The current frozen runner leaves features_*.txt empty, so the auditor **does not assume those empty files contain serialized features**. It reconstructs source features independently from raw source rows and compares native examples_train/dev/memory.txt.

WordPiece uses the 30522-token base BERT vocab and standard special IDs0/100/101/102/103. Direct BertTokenizer construction calls sanitize_special_tokens to match from_pretrained's special-token registration without adding vocabulary IDs. An initial CPU test caught the missing registration in the first auditor draft; that auditor-only defect was corrected before the passing checks below. No native code or running job was changed.

Entity IDs follow the ordered first column of FB entity2textlong.txt / WN entity2text.txt, with the same native description sanitization. Relation token IDs follow relations.txt order (FB237/WN11), **not** every relation2text entry (FB has1345). HF added-token offsets and total model vocabulary size are independently checked.

Reconstruction implements the native corruption/head-or-tail target selection, textual [MASK]/[PAD] templates, longest_first pair truncation64, entity/relation [PAD] placeholder replacement at offset30522, and dynamic pad-to-multiple8 per observed batch. It validates target index, full padded input IDs, exactly one mask and all token ranges. Condition-token tensors are absent from the JSONL and are not claimed independently verified; their frozen source is byte-bound.

## Exact exported exposure and metric arithmetic

Expected final exposed edit counts:
- FB: **3086/3087**; native drop_last omits zero-based row3086.
- WN: **830/830**.
- Every complete batch: locality8 then edit2, contiguous batch index/position. Total observations: FB15430, WN4150; locality exposures12344/3320.

Locality identities are independently reconstructed from NumPy default_rng(seed0).permutation(L-Test length). For worker4 round-robin batch dispatch, batch b uses permutation slice starting **8×floor(b/4)**. Each worker starts with the same cursor/permutation; repeated locality sources are expected. Exact candidate row IDs and observed multiplicities are checked/reported. This validates worker exposure lineage, not a full unique L-Test score.

Edit row candidates must equal the complete duplicate feature equivalence class, as the exporter preserves ambiguity. It does not invent a unique raw-row attribution for identical features.

Before/after ranks must be integers in1..entity_count; target logits finite; top3 three distinct in-range entity indices; rank≤3 must agree with target membership and exact top3 position. Full source-to-input/label lineage is checked.

The auditor independently accumulates:
- **Succ@1/3**: edit fraction with after rank≤1/3.
- **ER_roc**, v8 Eq.(4): |mean edit rank_after−mean edit rank_before| / mean edit rank_before.
- **RK@3**, v8 Eq.(3): count(locality after rank≤3) / count(locality before rank≤3).
- **RK_roc**, v8 Eq.(4): |mean locality rank_after−mean locality rank_before| / mean locality rank_after.
- Native Hits1/3/5, mean ranks, MRR, counts and unique-source counts.

Each recalculated field is compared with native-exposed-metrics.json using strict integer equality or tight floating tolerance. RK@3 is not silently substituted by locality Hits@3; baseline_hits3_count is explicit. These rank-change metrics are **not ROC AUC**.

Primary comparison targets from https://arxiv.org/html/2301.10405v8 Table2:
FB [.866,.986,.999,.874,.635]; WN [.833,.844,.991,.956,.256], in Succ1/Succ3/ER_roc/RK3/RK_roc order. Report includes descriptive deltas only, not automatic paper-match, SOTA or novelty assertions.

## Verification executed

Read-only live check after final code changes:
- Master6 + FB80 + WN86 = **172 checks; 0 failures**.
- Both **INCOMPLETE**, each missing training-complete, exposed metrics, terminal and exposed JSONL.
- Exit2, as intended for active runs.
- Canonical byte snapshot for that invocation:
  `9c4b43a5ac05768798c488c113104e72ebfe04b675ddf7d4f96c2cbd4c195af6`.
  This is observed evidence, not a pin that should remain constant as main updates runtime state.

CPU positive/negative tests used inline Python/in-memory StringIO/unittest mocks and wrote no fixture files:
- Independent source feature reconstruction using2 train +2 test +8 locality raw rows from each real dataset: **52 checks each; no failures**.
- Two exact synthetic batches with repeated locality and duplicate edit features: **378 checks; no failures**. Verified independent expected Succ1=.5, Succ3=1, ER_roc=.85, RK3=.5, RK_roc=0.
- Six separate mutations correctly rejected: wrong locality source (1 failure), wrong target(3), wrong top3(2), nonfinite logit(1), corrupted mask/source input(2), missing batch(4).
- CPU checkpoint/training-selection metadata fixture:45 checks,0 failures; abbreviated global-step count correctly rejected(1 failure).
- Final source AST parses successfully under the inspected Python parser.

These fixture results validate auditor logic, not these active runs' final observations or scores.

## Limits of an eventual PASS

**Native exposed evaluation**: edit_test participates in repeated validation and best-checkpoint selection. Do not describe this as untouched/pristine heldout. No split repair or novel heldout was created.

**Repeated locality exposures**: workers4 replaces author's32; different exposure weighting is recorded. Full unique L-Test inference is unavailable in this native export.

**Token-source contract checks, not all-score verification**: no model forward is run; JSONL omits full candidate score vectors. Therefore rank/top3 consistency and range checks cannot prove that every exported rank came from sorting all true model logits. Target logit finiteness does not prove score correctness. CPU tensor parity sidecar separately supports the observation adapter on a small fixture; this auditor does not upgrade that to full real-data score verification.

**Training evidence is bounded**: hashes, terminal/log binding, exact steps, metadata and scalar history support execution/selection consistency. This auditor does not independently replay the optimizer, infer models, time individual edits, or verify statistical superiority. Complete≠valid and an eventual PASS remains PASS_WITH_LIMITATIONS.
