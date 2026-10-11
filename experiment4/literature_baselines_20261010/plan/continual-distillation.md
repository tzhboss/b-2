# Continual KGE preparation handoff (2026-10-10)
Preparation evidence only; registrations, scheduler, environment and all GPU launches belong to the controller. Canonical status is ../experiments/KGE-LIT-001/experiment.yaml. No GPU job was launched by this preparer.

## First ready native full task: IncDE / ENTITY
Official repository association: AAAI publisher PDF https://ojs.aaai.org/index.php/AAAI/article/download/28722/29396, printed p8763 / PDF page5, explicitly links https://github.com/seukgcode/IncDE. Commit 46863500850bb1324db278743ba42335514e3032.
Data: repos/IncDE/data/ENTITY/{0,1,2,3,4}/{train,valid,test}.txt plus author precomputed train_nodes_degree.txt, train_nodes_betweenness.txt, train_edges_betweenness.txt. Source archives data1.zip SHA256 930df931dec885cae2401ecf1cdedcf2320734cce73ae143cfd22e9065019127 and data2.zip SHA256 a0a3b5c94e826f825bee83fa76470303f73c5bca9f29866c24b568e0313007e3. All extracted files preserve these immutable upstream ZIP originals.
Prerequisites: existing Torch2.5.1+cu124/Python3.12, numpy, prettytable (and its wcwidth dependency). Removed unused torch_scatter import; AST inspection found zero scatter function calls anywhere in IncDE. No package was installed by preparer. Do not install the author's entire environment export or downgrade Torch; main owns all env setup.
Compatibility/observation diff: repos/IncDE/provenance/compat-observation.patch; exact original source copied to repos/IncDE/provenance/original_source/. No method/loss/default epoch/split changes.
Native defaults: seed3407, TransE float64 dimension200, Adam1e-4, batch1024, margin8, neg10, five snapshots, 200max epochs per snapshot, patience5 with stop only epoch>30. Two-stage20 setting; embedding() freezes old rows while epoch<=20, multilayer loss starts at epoch20. Default layer weight .01 multiplied by10 on final snapshot. Keep these author details rather than silently correcting them.
Author main generates/reorders hierarchical files using provided centralities; DO NOT run data_preprocess.py: it recalculates expensive exact centrality, includes a RELATON typo, and bundled results already supply inputs.

Controller command (set ROOT to /data/lc/tzh/experiment4/literature_baselines_20261010, PYTHON to /data/conda/envs/test/bin/python, ASSIGNED_GPU only after assignment, RUN_BASE to a fresh absolute directory with existing parent):
```bash
cd "$ROOT/repos/IncDE"
test ! -e "$RUN_BASE" || exit 1
mkdir -p "$RUN_BASE/checkpoint" "$RUN_BASE/logs" "$RUN_BASE/ranks"
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" PYTHONHASHSEED=3407 "$PYTHON" -u main.py \
  -dataset ENTITY -gpu 0 -data_path "$ROOT/repos/IncDE/data/" \
  -save_path "$RUN_BASE/checkpoint/" -log_path "$RUN_BASE/logs/" \
  -rank_export_dir "$RUN_BASE/ranks"
```
This runs full native training and all snapshot evaluations, not a short smoke run. Use fresh RUN_BASE: author prepare() contains rmtree on existing method checkpoint folder. No existing checkpoint folder is deleted by the preparation above.

Ranks: patched evaluator exports head/relation/tail integer IDs, integer rank BEFORE float conversion, candidate count, filtered-count, train/test snapshot IDs and seed. 15 post-train task-matrix files plus4 pre-FWT files expected. Files refuse overwrite. Author checkpoints store state_dict+epoch_id at every epoch and best at each snapshot (0model_best.tar through4model_best.tar). Per-query exports independently recompute raw MRR and Hits; author evaluator rounds each snapshot metric to4decimals then query-count-weights task results and rounds overall to3decimals. NOT an equal-task macro average despite loose paper prose.
Evaluation contract: bidirectional via inverse relations; tail candidates and filter truths are historical QUERY snapshot, even when evaluating after later training. Filter train+valid+test up to that query snapshot; restore target score; double argsort ranking, no tie average. Do not compare directly to all-final-entity or targeted edit protocols.
Completion requires all5 trained snapshots,19 rank files,5 best checkpoints and terminal successful process; independent rank-count and metric audit required. No result/reproduction success is claimed now.

## Other official sources under preparation
LKGE: https://ojs.aaai.org/index.php/AAAI/article/download/25539/25311 (paper p4221 / PDF page5) directly links https://github.com/nju-websoft/LKGE. Commit5a74daefb72cf9718f604b0f29d277dbc638a20d. dataset.zip SHA2569c77b3f021d2b9c6f8138ee38aed3a4ac5f698a2eaf6ccae151f9d35b33b36c2, four complete5-snapshot datasets unpacked. Important source defect: src/data_load/KnowledgeGraph.py:27 loads test.txt as validation. Original author-default CLI also has using_embedding_transfer=False. Must distinguish source-native contaminated selection from paper-protocol clean adaptation. Do not silently change split then label exact author-code reproduction.
SAGE: arXiv2508.11347v1 paper PDFpage2 cites DOI10.5281/zenodo.15520114; author release https://zenodo.org/records/15520115 by paper coauthor Hang Yan links https://github.com/lyfxjtu/Dynamic-Embedding/tree/V1.0. GitHub association VERIFIED through paper-linked release, not guessed. Commit/tagV1.0 d40a6031c8a1eb8dae15454b9b4da7ac2c41c7f9. Native data for7 datasets in repo. Preliminary Softmax warning was superseded by the explicit CPU witness below; preserve default sampling. Second full native command is now prepared below.

## Second ready full native task: SAGE / FACT (author V1.0 defaults)
Status of preparation: source parses and imports on CPU with the downloaded, ABI-matched wheel artifacts. This is NOT training success or a completed reproduction. All seven author datasets are present; FACT is the actual default. No GPU launch/environment/canonical mutation was performed.
Frozen source: d40a6031c8a1eb8dae15454b9b4da7ac2c41c7f9, exact V1.0 release linked by the paper's Zenodo concept DOI. Native split loader reads valid.txt correctly. Defaults remain seed55, dimensions[180,210,250,270,290], FACT, max200epoch/snapshot, Adam1e-4 batch2048, margin8 neg10, patience3, mapping-training1epoch, replay setting1000, use_difficult_samples=False, LKGE backend label and original regularization/reconstruction behavior (FACT reconstruct weight1). Formula flag only enters log filenames; dimension schedule is explicitly predefined. Do not claim the released code executes the paper's scale-fit or dynamic-distillation formulation: source inspection found no dynamic-distillation implementation and get_dim_list directly uses parser values.
Correction to preliminary warning: CPU witness confirms torch.nn.Softmax(target_logit).dim.tolist() RETURNS target_logit.tolist() (the constructor stores its dim argument). It does not apply softmax and does NOT raise. Preserve author positive-score normalization unchanged. No sampling/loss fix was applied. Coverage loop has an author unconditional break; preserve it, document code/paper mismatch rather than inventing an algorithm.
Observation/output-only edits: repos/SAGE/provenance/observation.patch, originals in provenance/original_source. Add optional -rank_export_dir for normal/pre-FWT/post-expansion query ranks; optional -result_export_dir routes the author's terminal JSON to fresh output. No loss, sampling, epoch, dimension or split changes.
Dependencies downloaded as artifacts only: prettytable3.7.0 + wcwidth0.2.6 under repos/IncDE/provenance/dependency_wheels; quadprog0.1.13 and torch_scatter2.1.2+pt25cu124 CPython3.12 Linux under repos/SAGE/provenance/dependency_wheels. CPU import and scatter_mean ABI checks PASS with Torch2.5.1+cu124. Source-wheel inspection directories listed in download_manifest.json. Controller can install these wheels into its selected environment OR use a process-local PYTHONPATH over their extracted contents; preparer did neither persistently.

Controller command (ROOT/PYTHON/ASSIGNED_GPU/FRESH RUN_BASE as above; this process-local dependency path requires no shared environment install):
```bash
cd "$ROOT/repos/SAGE"
test ! -e "$RUN_BASE" || exit 1
mkdir -p "$RUN_BASE/checkpoint" "$RUN_BASE/logs" "$RUN_BASE/ranks" "$RUN_BASE/json"
SAGE_WHEELS="$ROOT/repos/SAGE/provenance/wheel_contents"
PYTHONPATH="$SAGE_WHEELS/torch_scatter-2.1.2+pt25cu124-cp312-cp312-linux_x86_64:$SAGE_WHEELS/quadprog-0.1.13-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64:$ROOT/repos/IncDE/provenance/dependency_wheels/prettytable-3.7.0-py3-none-any.whl:$ROOT/repos/IncDE/provenance/dependency_wheels/wcwidth-0.2.6-py2.py3-none-any.whl${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" PYTHONHASHSEED=55 "$PYTHON" -u main.py \
  -gpu 0 -data_path "$ROOT/repos/SAGE/data/" \
  -save_path "$RUN_BASE/checkpoint/" -log_path "$RUN_BASE/logs/" \
  -rank_export_dir "$RUN_BASE/ranks" -result_export_dir "$RUN_BASE/json/"
```
No -dataset override: native default FACT. 15 post_train +4pre_FWT +10post_expand rank files expected, plus5best checkpoints and2terminal JSONs. Normal task-matrix metrics have the same historical query-snapshot candidates/filter and sample-weighted aggregation as IncDE. Post-expansion results are separately labeled, not substituted for final results.
Cost ceiling: 37,000 main optimizer steps at200epochs each of5snapshots; early stopping may reduce this. Additional4mapping epochs, difficulty/score sampling passes and post-expansion evaluations occur. Max100x14541x290 float64 broadcast is about3.37GB for a SINGLE intermediate;24GB GPU is plausible but peak allocation and actual wall time are UNMEASURED. RTX3090 float64 throughput is low; do not quote a benchmark duration before running. OneGPU/task. Original Sum_Training_Time omits replay generation, mapping, tests, preprocessing and reload, so controller should record full-process wall time independently.

## LKGE preserved native vs clean split commands
Both variants retain source defaults: ENTITY-selected comparison, TransE dimension200, Adam1e-4, batch2048, margin8, neg10, max200epochs/snapshot,5snapshots, seed55, patience3, alpha.01/beta.1; using_embedding_transfer remains False. Native validation is test.txt (test-selected checkpoints), while clean variant reads the distributed valid.txt. Changing validation also changes encounter-order IDs/filter coverage; profiles are explicitly different. Neither profile is registered here.
Freeze: repos/LKGE/provenance/frozen_split_profiles.json; clean diff repos/LKGE/provenance/paper-clean-split.patch SHA25680de3bd9e5a334352e9d755637f29afa6173324653bbfbd86ed3c171fc53dbc2. Pristine files remain recoverable from exact Git commit, archives, and original_source. Native observation-only diff separately saved.
If controller selects a LKGE task, choose cwd explicitly:
```bash
# Native: "$ROOT/repos/LKGE"; clean adaptation: "$ROOT/repos/LKGE/clean_split"
cd "$LKGE_PROFILE_CWD"
test ! -e "$RUN_BASE" || exit 1
mkdir -p "$RUN_BASE/checkpoint" "$RUN_BASE/logs" "$RUN_BASE/ranks"
# NATIVE_DEPS is the same wheel-artifact PYTHONPATH string shown for SAGE.
PYTHONPATH="$NATIVE_DEPS" CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" PYTHONHASHSEED=55 \
"$PYTHON" -u main.py -dataset ENTITY -gpu 0 \
  -data_path "$ROOT/repos/LKGE/data/" -save_path "$RUN_BASE/checkpoint/" \
  -log_path "$RUN_BASE/logs/" -rank_export_dir "$RUN_BASE/ranks"
```
Native or clean ENTITY main training ceiling37,000optimizer steps. Native LKGE provides19rank exports and5best checkpoints; clean profile exports same file contract. Source-native is explicitly ineligible as clean heldout evidence. Clean profile is a split-adapted author-default baseline, not an exact paper-table reproduction.

## Independent metric/query-contract audit
```bash
"$PYTHON" "$ROOT/code/prepare_continual_distillation.py" --root "$ROOT" \
  --method IncDE --dataset ENTITY --audit-ranks "$RUN_BASE/ranks"
"$PYTHON" "$ROOT/code/prepare_continual_distillation.py" --root "$ROOT" \
  --method SAGE --dataset FACT --audit-ranks "$RUN_BASE/ranks"
# For LKGE: --method LKGE --dataset ENTITY --split-profile clean OR native.
```
The helper uses independently reconstructed IDs/query order/filter sets from author text, validates complete query counts and integer rank ranges, then computes raw float64MRR/Hits and author round4 -> weighted -> round3 aggregates. Missing expected exports return nonzero exit. This is independent metric aggregation; it does not alone verify the original model scores. Best checkpoints permit separate score re-evaluation for post_train; SAGE original code does not persist post-expansion model states, so post_expand exports are auditable for metrics but full checkpoint-based score re-evaluation needs an additional observation checkpoint save if requested.
CPU patch checks: all original/patched metrics equal with export disabled/enabled on7queries across2batches; SAGE post_expand separately passes. Native FACT loader agrees exactly with independent IDs/query order across all5snapshots (24,810bidirectional test queries each); initial actual native model CPU prediction is finite. These are preparation checks, not training/benchmark results.
Do not rerun preparation over active IncDE datasets: native generate_layers may rewrite sorted auxiliary files. The script now supports --methods LKGE SAGE for safely continuing preparation outside active IncDE source/data.
CrossRef title+DOI match verified for all3. Semantic Scholar IDs: IncDE f42d060fb530a11daecd90695211c01a5c264f8d; SAGE f1833b793c9c7f72af775e59495e8afae945ca6b. LKGE API lookup had a DNS failure; primary publisher PDF plus CrossRef directly verify it, and this fallback is retained in source_verification.json.
Primary PDFs, text extraction, author archive/API metadata, exact commits/tags, all dataset/snapshot file SHA256, source hashes, dependency wheel URLs/checksums, defaults and computational ceilings are saved per repository under provenance/. No unaudited evidence was promoted to results/, no canonical metadata or scheduler/environment file was edited, and no job was launched by this preparer.
