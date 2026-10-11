# FastKGE / STARK independent audit

Read-only CPU CLI is ready. Controller owns reruns, integration, canonical, scheduling, GPU allocation and promotion. Worker writes are the assigned audit script and this review only. No GPU jobs or shared environment changes.

## CLI

From /data/lc/tzh/experiment4/literature_baselines_20261010:

    /data/conda/envs/test/bin/python code/audit_fast_stark_independent.py --root /data/lc/tzh/experiment4/literature_baselines_20261010 --method FastKGE --threads 2
    /data/conda/envs/test/bin/python code/audit_fast_stark_independent.py --root /data/lc/tzh/experiment4/literature_baselines_20261010 --method STARK --threads 2

JSON to stdout; controller chooses retention path. Exit0/1/2 = passed/failed/incomplete. Default replay-samples=0 gives PASS_WITHOUT_CHECKPOINT_REPLAY only after all19 exports, five best selections/checkpoints and terminal evidence. Add --replay-samples 8 for120 CPU queries across15 post-training files. Optional --run overrides the run directory. No hard-coded host paths inside script, no output writes or CUDA device use. Default runs: root/runs/{fastkge,stark}-fbckge-s3407/native/seed3407.

## Observed evidence

Initial FastKGE CPU audit without inference:295 checks, zero failures/missing,19 exports and five best selections. Final raw weighted MRR0.22255063358641675, H1 .13110036275695283, H3 .25635630794034664, H10 .40199919387343813. Native rounding yields .223/.131/.256/.402, matching log. Native training449.15384817123413s. Worker-read process_exit.json reported572.3322098989738s; controller's outer process timing may have a different boundary.

STARK snapshot:9 exports, zero failures among209 checks, later tasks/checkpoints/terminal evidence missing, INCOMPLETE. This evidence snapshot is not duplicated runtime status; current status remains controller canonical.

Worker development replay60/60 ranks exact across15 files, independent/native score difference0, but three state assertions failed. Controller retained logs/FastKGE-independent-audit-attempt1.json:346 checks, FAIL3 restored_native_state_2/3/4,120/120 ranks exact, score difference0. Attempt1 is preserved, SHA256 fae485a99c845690f7283cced420df9fa6fae08762b3226141780c133b098a9f. Exact ranks did not erase incomplete-state failures.

## Reviewed reconstruction correction

Actual snapshot2/3/4 dictionaries each contain six extra top-level buffers: old_data_lora_rel_embeddings_lora_A/B and old_data_lora_ent_embeddings_list_{0,1}_lora_A/B. Snapshot1 has none. Native store_old_parameters() registers previous trainable A/B as buffers; loralib.lora_state_dict() includes them because their names contain lora_. The prior audit registered only old entity/relation prediction buffers and current LoRA modules. Those saved previous-LoRA buffers therefore appeared as unexpected keys. This was a real omission in the audit object.

Repair registers every saved old_data_* buffer before loading. The strict assertion remains: no unexpected keys; missing keys exactly the two old entity/relation prediction buffers rebuilt from previous best states. New checks require every persisted tensor to match after load/eval and previous-LoRA buffers to match previous best checkpoint A/B exactly. No keys are ignored to force pass.

Isolated CPU state test, without rerunning controller's audit: all five checkpoints restored all saved tensors exactly;11 checks, zero failures, including three previous-LoRA lineage checks. Full repaired audit rerun/integration remain controller-owned. Repaired script SHA25603ddc4d3da7aeee2163eb0b009ae168adc050b27e2039e4c51cdcfe3c8a6b504. Pre-repair observed SHA256095366f64413aa323ef559004b17345da4b2093a5cb73bf83b78fe69c22fc689.

Scope: all checkpoint-covered LoRA state/buffers plus inference-required old entity/relation buffers. This is not exact training-resume state. Expanded random base caches, optimizer state and increment initialization RNG were not saved. Those random caches do not enter native predict(), which reads old_data entity/relation and current LoRA modules. Complete optimization-process reconstruction is not claimed.

## Independent data / metric contracts

The script does not call preparer audit_inputs.py/audit_ranks.py. It scans snapshots0–4 and train→valid→test, assigns entity IDs by head/tail encounter and consecutive forward/inverse relation IDs. Rebuilt mappings and train_id.txt must match every author file. It hashes raw splits/IDs and checks split overlaps and cross-snapshot duplicates.

Test facts are stably grouped by ascending forward relation ID. Queries are all forward facts then all inverse/head-prediction facts. Every NPZ query sequence must match exactly. Coverage:15 after files train i/test0…i, plus4 before FWT train i/test i for i1…4.

Actual labels/filter tensors use cumulative train+valid+test through TEST snapshot; the local training-snapshot hr2t variable in DevBatchProcessor is unused. Candidates are IDs0…test_snapshot.num_ent−1:7505/11258/13134/14072/14541;474 relations throughout. Targets must be in rebuilt filters; ranks integral and within candidates.

Native normalized TransE scores sigmoid(9−L1(normalize(h)+normalize(r)−normalize(candidate))). Filter positives to−10000000, restore target score. Rank1+double torch.argsort preserves author ties. Native float32 reciprocal sums batch100 and4-decimal task rounding are checked against independent float64 NumPy MRR/Hits. Overall weights test-size averages of rounded tasks, then3-decimal rounding. Full task log tables, overall log, BWT/FWT and training time sum are checked. FWT uses four before MRRs; BWT final-minus-diagonal MRR over tasks0–3.

Five selections require metadata identities, epoch<200, best-file SHA equal selected epoch-file SHA, first strict validation-log maximum, finite tensors and checkpoint0 epoch_id. Feature ordering, group length and adaptive ranks are rebuilt from bundled structural files and checked against LoRA shapes. Input/source/launch/rank/selection/checkpoint hashes are emitted.

## Original source / observation equivalence

Pinned FastKGE777b3d88f2adf3a47cc773072e1fc53aba558f38; STARK2e546b30688c660903247df89330671f4f73bf68.

Only main.py, src/parse_args.py, src/model/BaseModel.py and src/model/model_process.py may differ. git diff must exactly match preserved reproduction/compatibility.patch; prepared source manifest is checked. Original/current AST equality after removing only observation statements, allowed numeric parser types and unused torch_scatter import is required. Model/loss/data/train/test/util ASTs otherwise unchanged; main.sh text unchanged.

Independent fixture executes original HEAD evaluator and patched evaluator disabled/enabled: seven queries, batches4+3, exact-score ties. Metrics and query/rank arrays agree. Export filesystem calls intercepted in memory; zero fixture disk writes. This checks observation equivalence, not independent retraining.

## CPU sampled inference

Validation leaves loralib embeddings eval/merged=True; save_model occurs after validation. Since module names contain lora_, checkpoint includes .weight as well as A/B. Saved .weight already includes BA/r. Adding delta again would corrupt inference.

Independent assembly appends saved merged group weights to previous assembled best embeddings and applies author's actual structural-order indexing, retaining surprising original permutation semantics and padding behavior. Old prediction buffers follow previous-best lineage. Zero incremental relations and empty relation modules are preserved.

A second path restores native class definitions and loralib objects in memory, initializes Embedding metadata, scaling1/r, merged=True and saved buffers, and loads every persisted key. No main/parser/random constructor runs. Independent/native full-candidate scores must agree. Rebuilt filters and double-argsort independently compare exported ranks.

Sampling evenly spans both query halves; exact ranks and exact-score tie-compatible intervals are separately counted. This is sampled, not all-query inference. Four before files receive full query/metric checks but no replay because initial new-layer weights/RNG were not saved.

## STARK embedding lineage / method boundary

Native run() calls switch_snapshot(), then saves repos/STARK/embeddings/FB_CKGE/{snapshot}/{entity,relation}_embeddings.npy outside run directory. Snapshot0–3 contain next random expansion; snapshot4 final assembled embeddings. Audit inventories hashes/shapes/dtypes/mtime at read time only. No moves/rewrites or substitution for checkpoint reconstruction. While running these paths can appear/change; future native runs may overwrite them. Controller should retain finished-run hashes.

Released STARK uses second column of train_distance_nodes2.txt (distance), not PageRank. Preserve published-code semantics; full-paper equivalence/targets remain unverified. Native split execution is distinct from separately frozen clean comparison/edit splits.

## Primary FastKGE paper proof and targets

Official pinned PDF repos/FastKGE/reproduction/sources/paper.pdf SHA2568417ff7120f40ed8eb1acdc1fb88343140a822c0314b2c8617a6c9c1725b110c.
https://www.ijcai.org/proceedings/2024/0243.pdf ; DOI10.24963/ijcai.2024/243.

PDF page5/published2202 explicitly states “The datasets are available at https://github.com/seukgcode/FastKGE.” Direct paper→author repository proof. Pinned fulltext confirms5 snapshots,3:1:1 split, embedding200, Adam, patience3, RTX3090Ti and five random runs.

Table3 PDF page6/published2203, columns HYBRID/FB-CKGE/WN-CKGE: FastKGE FB-CKGE MRR .223, H1 .131, H3 .257, H10 .405, training405s. Read directly from pinned PDF. Single-seed .223/.131/.256/.402 is not the five-run paper mean; RTX3090 and timing boundaries differ. No clean edit performance/SOTA/five-seed reproduction claim follows.

Previously invoked deep-research quick verification: primary-paper proof, pinned provenance, explicit protocol and verification limits.
