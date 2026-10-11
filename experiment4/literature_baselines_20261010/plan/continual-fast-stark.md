# Continual FastKGE / STARK preparation handoff

Prepared 2026-10-10 with deep-research quick source-verification. Controller exclusively owns registration, canonical, scheduling, GPU allocation and promotion. This file describes preparation evidence, not canonical experiment status. No GPU job was launched. No shared Python/download environment was altered. All writes stayed in the two owned repositories and this handoff.

## Ready full native training commands

Run these independently in controller-managed jobs. Controller must assign distinct physical GPUs through FASTKGE_GPU and STARK_GPU. The wrapper maps each card to the native logical -gpu 0 and refuses an existing run-name directory.

```bash
cd /data/lc/tzh/experiment4/literature_baselines_20261010
/data/conda/envs/test/bin/python repos/FastKGE/reproduction/run.py --gpu "${FASTKGE_GPU:?controller must assign GPU}" --run-name native-seed3407
/data/conda/envs/test/bin/python repos/STARK/reproduction/run.py --gpu "${STARK_GPU:?controller must assign GPU}" --run-name native-seed3407
```

Append --dry-run to inspect expanded argv without launching. No conda activation or installation command is needed. Launchers use config-relative paths, the supplied interpreter and four CPU math threads each. Their exact native argv is in each reproduction/config.json.

After full completion:

```bash
/data/conda/envs/test/bin/python repos/FastKGE/reproduction/audit_ranks.py --run-name native-seed3407
/data/conda/envs/test/bin/python repos/STARK/reproduction/audit_ranks.py --run-name native-seed3407
```

This separately recomputes metrics from raw query ranks and validates query/checkpoint coverage. Controller must independently audit before promotion.

## Author dataset and full settings

Both use bundled author data/FB_CKGE, all five snapshots 0–4, original train/valid/test files and precomputed structural features. The 15 raw split files are byte-identical across repositories. Cumulative ID mappings and every train_id file were independently rebuilt and matched exactly. There are zero within-snapshot split overlaps and zero triples repeated across snapshots.

| Snapshot | Entities | Train | Valid | Test |
|---|---:|---:|---:|---:|
| 0 | 7505 | 111642 | 37214 | 37214 |
| 1 | 11258 | 18607 | 6202 | 6203 |
| 2 | 13134 | 18607 | 6202 | 6203 |
| 3 | 14072 | 18607 | 6202 | 6203 |
| 4 | 14541 | 18606 | 6202 | 6202 |

All snapshots have 237 original relations / 474 with inverse. Input manifests contain per-file SHA-256 and row counts. Common raw-split manifest SHA-256: c9bbcd94e8d93dbb6b8eaad9c06874ebe0a64818218c5597b574f115c1f3a50c.

| Setting | FastKGE | STARK |
|---|---|---|
| Epoch ceiling per snapshot | 200 | 200 |
| Native validation-MRR patience | 3 | 10 |
| Seed / batch / embedding dimension | 3407 / 1024 / 200 | 3407 / 1024 / 200 |
| Adam learning rate / negatives / margin | 0.1 / 10 / 8 | 0.1 / 10 / 8 |
| Entity rank / relation rank / entity groups | 150 / 20 / 2 | 200 / 200 / 20 |
| Adaptive ranks / relation groups | True / 1 | True / 1 |
| STARK additional flags | none | lambda_dyn=1.0, fusion_weight=0.2, hub_scale=0.05 |

Values follow author main.sh plus parser defaults. No model, loss, splits, precision, negative sampler, evaluator tie policy or epoch ceiling was changed. Full author early stopping remains enabled. No one-epoch smoke is presented as reproduction.

## Sources and verification boundary

FastKGE: Jiajun Liu et al., Fast and Continual Knowledge Graph Embedding via Incremental LoRA, IJCAI 2024, pp.2198–2206. https://doi.org/10.24963/ijcai.2024/243 ; official code https://github.com/seukgcode/FastKGE . Official PDF https://www.ijcai.org/proceedings/2024/0243.pdf was downloaded and read; Crossref independently matched metadata. PDF p.5 confirms Adam, dimension 200, patience 3 and five random runs. PDF p.6 Table 3 gives FB-CKGE 405 s total training on RTX3090Ti. Current launch is one complete native seed instance, not the paper five-run mean; the exact published five-seed set was not found. Paper settings and existence verified; performance not reproduced yet.

STARK: Kyung-Hwan Lee and Dong-Wan Choi, STARK: Structure-Aware and Adaptive Representation Learning for Continual Knowledge Graph Embedding, Proceedings of the ACM Web Conference 2026. https://doi.org/10.1145/3774904.3792391 ; official code https://github.com/bigdata-inha/STARK . Crossref matches title, authors, Inha affiliations, venue, and publication date 2026-04-12. Semantic Scholar ID 9a5b0c50be2d6f9e792613bec4357be5b9a9f83b independently matches title/year/authors/DOI. Full paper unavailable: S2 openAccessPdf CLOSED; publisher PDF HTTP403. Full-paper settings/numerical targets remain unverified. Report a released-implementation execution, with the semantic caveat below.

Search strategy: user-supplied official repos, exact-title searches, DOI/Crossref/S2, then README/main.sh/parser/data loader/model/evaluator/preprocessing inspection. FastKGE S2 timed out; official PDF + Crossref supplied verification. Direct browser DOI/DBLP opens failed and were not counted as verification. Skill evidence grade: FastKGE controlled benchmark paper level III; code evidence/author descriptions gray literature level VII for unverified performance claims. STARK paper existence passes; full methodological assessment is unavailable. These are conference proceedings, so journal-specific indexing/predatory-journal checks are inapplicable. Author code is not independent performance corroboration. AI-assisted source verification/preparation was used.

## Minimal compatibility and observation-only changes

Each repo reproduction/compatibility.patch preserves the tracked diff against its actual Git HEAD. Only four tracked source files changed: remove unused torch_scatter import (no scatter calls exist), type numeric CLI arguments without changing defaults, add optional per-query Test-rank export, and record selected best checkpoint epoch/validation metric. Original ranking/filtering and model/loss calculations are untouched.

Original code already saves best checkpoint per snapshot; snapshot 0 contains full state_dict, later ones contain author lora_state_dict. Later native checkpoints omitted selected epoch metadata; selection JSON now records it. predict_result originally wrote only top-three entity names, not per-query ranks. Export files contain head/relation/tail/rank, candidate count, training/test snapshot and before/after phase. Expect 15 post-training rank files and 4 pre-training FWT files. Independent NumPy recomputation checks all 19 evaluations against exact raw test-query order, recomputes MRR/Hits and BWT/FWT, and hashes selected checkpoints. This is independent metric recomputation from exported ranks, not independent checkpoint inference; later native LoRA-only checkpoints may require state reconstruction. No checkpoint format was changed.

Dependencies were installed only into each owned reproduction/vendor using pip --target --no-deps: loralib==0.1.2, prettytable==3.7.0, wcwidth==0.2.13. Shared Python3.12 / Torch2.5.1+cu124 / NumPy2.4.0 are reused. Do not install upstream requirements.txt into the shared environment: it pins old Python3.7-era versions and FastKGE contains author-local wheel paths. Bundled features mean no matplotlib/network preprocessing dependencies are needed.

## Preparation evidence

Both real-data CPU API preflights passed across all five snapshots: native forward/backward/predict and snapshot transitions, with zero optimizer steps. FastKGE 33.352 s; STARK 34.801 s. These are CPU API timing observations, not training-runtime or paper reproduction measurements. Both synthetic observation tests exited 0: export enabled/disabled ranking and metrics identical, expected ranks [1,1,3] recovered, and full/LoRA best-checkpoint selection metadata checked. An initial helper cleanup failed on NFS because an NPZ was still open; explicit close fixed it and successful reruns were observed. No empirical KGE MRR/Hits results exist yet.

Files under each reproduction/: config.json, run.py, requirements-minimal.txt, vendor/, compatibility.patch, preflight_cpu.py/json, check_observations.py, observation_check.json, audit_inputs.py, input_manifest.json, audit_ranks.py, source_manifest.json, prepared_source.tar.gz, source_verification.json and sources/. The source archive captures exact prepared code/config/dependencies; hashes provide tamper-evident evidence. Source manifests list per-file SHA-256; raw data stays in the released repo with input hashes. No dataset/checkpoint/output commits were made.

A full launch creates reproduction/runs/native-seed3407/launch.json, stdout.log, native logs/checkpoints, observations and process_exit.json. Those are raw evidence and are not automatically promoted. Existing run directories are refused to preserve prior attempts. Controller owns canonical status and final audit.

## Scheduling estimates and limitations

Planning estimates only, no GPU benchmark performed: FastKGE 15–120 min and STARK 30–240 min per complete five-snapshot seed. FastKGE paper timing is training-only 405 s on RTX3090Ti; current hardware/runtime, file I/O, and final evaluations differ. STARK paper runtime is inaccessible. Its released distance-column allocation produced ranks up to 2691 in CPU preflight, so the larger allowance is prudent. Keep full 200-epoch ceilings and author early stopping; do not truncate into a smoke experiment if estimates are exceeded.

Reserve one RTX3090 per method; plan 8–16 GiB GPU peak on each 24 GiB card (actual unmeasured), four CPU threads, 8 GiB host RAM and 20 GiB output allowance each. Native evaluation batch=100, dimension=200, float64 materializes up to 100×14541×200 differences (~2.17 GiB per tensor before gradient/workspace). Every-epoch native checkpoint writes can consume several GiB. Cloned repos currently occupy 331 MiB FastKGE and 572 MiB STARK. Preparation inventory saw seven idle 3090 cards and 171 GiB host RAM available then; that is not a current reservation. Controller must recheck and assign around its other jobs.

Execution readiness: both native training entries passed CPU preparation. GPU execution/peak memory remain untested here because no launch was authorized for this worker.

STARK methodological blocker: cal_features.py active entry calls solve_network_with_betweenness_weighted_pagerank without required data_name. Therefore launch directly consumes author-bundled precomputed features and skips preprocessing. nodes_sort.py emits ID/distance/PageRank, but released model reads column 2 (distance) as score and ignores PageRank. Preserve this published-code behavior, do not silently “fix” semantic choices or assert paper-method equivalence. FastKGE bundled feature files omit 9/10/5 IDs at snapshots 2/3/4; native zero fallback is preserved. Snapshot0 features are intentionally empty and are not used for an increment. STARK unused all_triples at snapshot4 is cumulative; native loader uses only train/valid/test. Neither baseline is directly comparable to single-edit KGE without separately registered adaptation.

## Exact immutable preparation identifiers

### FastKGE

- Upstream commit: 777b3d88f2adf3a47cc773072e1fc53aba558f38
- Prepared-source manifest: f680d2efee93ffd17afa121c689afd2d29be9992ed7cc7155fcc34906ef0ce65
- Prepared archive SHA-256: 6c781fd63c998f3bdcafb2712d680a9fdfa79643a06f9a9a3e95a870f1ea51c4
- Compatibility/observation diff SHA-256: dc094345ebadc954a37dcdff03bf895e1b15da362168ad0722dac19d4dc9d02d
- Input manifest: fd9a8cb7194d54f211501e94d0a97a7f5ca9e31cbcc7fec3ca703e341d303de4
- Config SHA-256: 1c99b595f3ee9a9980bfc6d258c07fc45a883277346b1b6008168d761dc1a54f

### STARK

- Upstream commit: 2e546b30688c660903247df89330671f4f73bf68
- Prepared-source manifest: 9f1a71316b4002b640286f79feb34cd41d8d25ab783433fd37ddd479bd3d6b2f
- Prepared archive SHA-256: ef43549dbde698797ea13f1b19e08b45e2040d327d1478a1690bb9aec03cf06b
- Compatibility/observation diff SHA-256: c7e000a8774f42038e51a82109b304351fd70fbacbfd2ae6826a835899a7e86d
- Input manifest: 4dd0178f23fe2b5bb20bf1fc33e660c2e5a9b5388bf243932257cb41791e7d30
- Config SHA-256: cbaa470cfe071c52a811dcbe64cc78615176a1a92deeab5e74ef9dd5908ca111
