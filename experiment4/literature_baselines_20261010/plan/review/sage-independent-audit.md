# Independent SAGE native audit

Frozen CLI:
    /data/conda/envs/test/bin/python code/audit_sage_independent.py --root /data/lc/tzh/experiment4/literature_baselines_20261010 --replay-samples 8 --threads 2

Script SHA256 e594da28830bd4416869a54453d6b0b113940ad94b5c3b36d1c7eb1040d72d8f.
JSON stdout only; no environment/canonical/source/native-output writes. CPU only. Default replay-samples0 gives metric/source audit;8 checks120 queries across15 post_train files.

## Verification observed

825 scientific checks, zero failures/missing;29 rank files;5 best checkpoints;2 terminal result JSONs. Fifteen post_train files replayed on CPU:120/120 exact ranks and independent-vs-native scoring max difference0. Best epochs/dimensions: snapshot0 65/180,1 22/210,2 15/250,3 7/270,4 10/290. Native train-loop sum1510.4192135334015s. Final native table MRR .220, H1 .134, H3 .249, H10 .387. One seed55 is not the paper five-seed mean or a clean edit comparison.

Scientific status PASS_SAMPLED_POST_TRAIN_REPLAY; overall PRODUCER_INTEGRITY_FAIL_SCIENTIFIC_PASS, exit1 intentionally. Controller owns formal adjudication/promotion. No training rerun is needed to investigate the observed metadata drift; the original receipt is retained without edits.

## Original producer failure and bounded adjudication

runs/sage-fact-s55/terminal.json reports exit_code0 but status failed, source_inputs_unchanged false, source_error AssertionError('repos/SAGE/provenance/preparation_manifest.json'). It binds run.log SHA2567d23afb34f4b106bc2e77a1a2080f27d04c639b4c49b9cf819fffb44b443245d and elapsed1779.56712410599s. This original producer-integrity failure is not erased or turned into a clean pass.

Frozen master configs/experiments/KGE-LIT-003-jobs.json SHA2563d31b5a36b28aff0aa155d8fb1c26bef3b4831a22c53091a857d9c79c51a6165. All302 entries are checked:300 SAGE worktree copies and2 process-local external dependency wheels. Registered-job equals its exact master job; copied-source inventory equals the expected remap. Runtime argv differs only by the frozen repository→worktree data path; cwd/worktree and assigned GPU metadata match.

Exactly one ORIGINAL repository input drifted:
- repos/SAGE/provenance/preparation_manifest.json
- frozen/registered ba91fd8c79a3c1c3c37dbc2058489a9628da6252c44a04b1c357ab26b96420e2
- current7b9bf712841140cd93681c89131eeb22f7825ada380d017a5b1dbcd2b3a73a39
- worktree still matches frozen original hash.

Nine JSON leaf changes, emitted in full by auditor:
1. defaults/-rank_export_dir added null.
2. defaults/-result_export_dir added res_diff_datasets/.
3–6. resolved_url/content_type removed from two primary_sources records.
7–9. source_sha256 descriptions updated for main.py, src/parse_args.py, src/model/model_process.py, from pristine pre-observation hashes to already-frozen observation-patched source hashes.

No loss/model/split/raw-data/runtime-config file changed. Before hashes for those3 source descriptions equal pristine Git HEAD; after descriptions equal the frozen actual worktree files. The old manifest was stale bookkeeping relative to the master, which already bound patched source. No manifest reference/read appears in inspected native main/src Python code; manifest is not a runtime argument. Classification SUPPORTED_NONCONSUMED_METADATA_DRIFT. This is static-source/runtime-path evidence, not an OS syscall access trace. Controller must explicitly retain the failed receipt when recording any metadata-only adjudication.

## Independent contracts

Run runs/sage-fact-s55; executed worktree runs/sage-fact-s55/worktree; pristine commit d40a6031c8a1eb8dae15454b9b4da7ac2c41c7f9. Worktree code/data/dependency hashes bind to master. Complete effective log Namespace is checked against frozen parser defaults, FACT config, runtime paths and seed55. Effective process environment is not fully captured by runtime-command.json; frozen declared PYTHONPATH and binary/wheel files are bound, rather than inventing a runtime environment receipt.

IDs are rebuilt by snapshots0–4, train→valid→test encounter order; heads then tails; consecutive forward/inverse relation IDs. Test facts stable-sort by forward relation ID; all forward queries precede all inverse/head queries. Raw splits, mapping, query ordering and cumulative filters are hashed. Split overlap and cross-snapshot repeated triples are checked.

29 expected NPZs:15 post_train (i/test0…i),10 post_expand (i0…3/test0…i),4 pre_fwt (i1…4/testi). Exact query arrays, schema, integral ranks, seed/snapshot/phase/metric_contract metadata and candidate/filter counts are checked. Filters are train+valid+test cumulative through TEST snapshot. Candidate universe is historical query snapshot, not final/train snapshot. Filter count includes target; rank bound uses candidate_count-filtered_count+1.

MRR/MR/H1/H3/H5/H10 are independently calculated in float64 and separately with native float32 batch100 summation/round4. Raw-minus-rounded differences remain visible. Native task metrics, weighted overall round3, two result JSON contents, five task tables, final overall log, FWT/BWT and training-time sum are checked. Native train-loop timing excludes replay selection, mapping training and evaluations; full terminal wall time is separate.

Five best files must match selected epoch file hashes, epoch<200, finite float64 tensors, entity/relation dimensions180/210/250/270/290 and mapping shapes. Selected epoch matches first strict validation-MRR log maximum; native early stopping/full ceiling completion is checked. Validation ranks were not exported, so this is selection/byte-lineage consistency, not independent validation inference.

## Observation/source equivalence

Pristine/current Python AST comparison permits only export additions in model_process.py, two export CLI defaults in parse_args.py and result-directory routing in main.py. Other math, optimizer, loader, loss and training code must be unchanged. Originals retained in provenance/original_source. Preserved observation.patch is reproduced byte-for-byte with its actual Python difflib generation order; it is not mistaken for Git's differently formatted diff.

Seven-query CPU fixture in batches4+3, including exact-score ties, compares original versus patched disabled/enabled metrics and exported ranks/candidate/filter columns for post_train/pre_fwt/post_expand. Export writer is intercepted in memory; no fixture writes.

## Sample inference proof and exclusions

LKGE/TransE in this SAGE repository do not override BaseModel.embedding/predict/norm/post-processing. Normal BaseModel.embedding returns ent_embeddings.weight and rel_embeddings.weight directly. predict normalizes raw embeddings and scores sigmoid(9−L1(h+r−candidate)); no mapping or old_data buffer enters this post_train scorer. main.continual_learning loads selected checkpoint before post_train evaluations, then trains mapping afterward.

Thus saved raw selected embeddings are sufficient for independent post_train scoring. An independent assembly/formula path is compared with exact native scoring methods extracted by AST; no native main/parser/constructor/import side effects or GPU calls occur. Full historical candidate set, rebuilt filters, target restoration and double-argsort ranks are used. Eight evenly spaced queries/file include forward/inverse halves. Exact ranks and exact-score tie-compatible intervals are reported separately.

Ten post_expand files have full query/filter/count/metric audits but no exact checkpoint replay. Their mapping-training evaluation state was not saved at that phase; the next checkpoint's mapping is not assumed to represent that transient state. Four pre_fwt files likewise are not falsely assigned post_train checkpoints. No independent retraining or complete optimizer/RNG reconstruction is claimed.

Released fixed dimension schedule and replay normalization are preserved. Paper-method equivalence/dynamic schedule claims are not established. Original paper/release verification is separate from current artifact validity. User-requested quick source-verification workflow is applied through primary/pinned-source proof and explicit limitations.
