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
SAGE: arXiv2508.11347v1 paper PDFpage2 cites DOI10.5281/zenodo.15520114; author release https://zenodo.org/records/15520115 by paper coauthor Hang Yan links https://github.com/lyfxjtu/Dynamic-Embedding/tree/V1.0. GitHub association VERIFIED through paper-linked release, not guessed. Commit/tagV1.0 d40a6031c8a1eb8dae15454b9b4da7ac2c41c7f9. Native data for7 datasets in repo. Default replay sampling contains an invalid Softmax constructor use (.dim.tolist()); blocked for full default run pending scoped review. Prioritize runnable IncDE first.
