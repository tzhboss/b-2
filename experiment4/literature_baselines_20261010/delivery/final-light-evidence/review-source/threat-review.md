# Independent threat-source verification: shallow-KGE TRUE/Prediction QP + Rank fallback

**Date:** 2026-10-10. **Mode:** deep-research quick / source-verification sidecar. **Disposition:** actionable author-source options exist for Pons, AlphaEdit and AlphaEdit+; KBS metadata is verified but full-text/protocol/code verification remains incomplete. No novelty clearance or SOTA comparison is established.

## Scope and evidence status

I read remote /data/lc/tzh/AGENTS.md. Writes are confined to this report and ../papers/threat_review/. No GPU work, training, dependency installation, shared-environment modification, financial/audio work, or process-argument inspection was performed. Repository files were read as data, not imported or executed. The inherited paper_matrix/review were not used as evidence. Existing project prose was used only to locate the method; the QP contract below was checked against implementation source. This is not an audit of historical experiment metrics.

| Source | Primary identity/date | Full-text status | Author code status |
|---|---|---|---|
| Pons et al. | **Revisiting Catastrophic Forgetting in Continual Knowledge Graph Embedding**; Gerard Pons, Carlos Escolano, Besim Bilalli, Anna Queralt; arXiv:2604.19401v1, submitted **2026-04-21 12:26:56 UTC** | PDF and arXiv HTML retrieved/read | Paper §6.1.4 links RevisitingCKGE; pinned and inspected |
| Wang et al., KBS | **Structure-aware and parameter-efficient framework for editing LM-based knowledge graph embeddings**; Mingjie Wang, Jianhou Gan, Wenqi Ren, Jun Wang, Zijie Li, Juxiang Zhou; DOI **10.1016/j.knosys.2026.116976** | **FAIL_FULL_TEXT_RETRIEVAL**; metadata and publisher-indexed abstract/section excerpts only | **NOT_ESTABLISHED** in bounded search; not a finding of nonexistence |
| Fang et al., AlphaEdit | **AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models**; Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, Tat-Seng Chua; arXiv:2410.02355 v1 **2024-10-03**, v4 **2025-04-22**; ICLR 2025 proceedings | PDF and v4 HTML retrieved; equations/protocol read from HTML | Explicitly linked author repository pinned and inspected |
| Liu et al., AlphaEdit+ | **AlphaEdit+: Model Editing in the Presence of Conflicting and Inconsistent Knowledge**; Qing Liu, Jianhao Zhang, Ou Wu, Michael Ng, Yi Du; Findings ACL **July 2026**, pp. **14809–14835**, DOI **10.18653/v1/2026.findings-acl.728** | Official 27-page PDF retrieved/read | PDF p.14809 links AlphaEdit_plus; pinned and inspected |

Primary URLs: [Pons record](https://arxiv.org/abs/2604.19401), [Pons PDF](https://arxiv.org/pdf/2604.19401v1), [KBS DOI](https://doi.org/10.1016/j.knosys.2026.116976), [publisher](https://www.sciencedirect.com/science/article/pii/S0950705126017028), [Crossref deposit](https://api.crossref.org/works/10.1016/j.knosys.2026.116976), [AlphaEdit record](https://arxiv.org/abs/2410.02355), [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html), [AlphaEdit+ record](https://aclanthology.org/2026.findings-acl.728/), [AlphaEdit+ PDF](https://aclanthology.org/2026.findings-acl.728.pdf).

**KBS date reconciliation:** publisher-indexed material says available online **5 September 2026** and displays a pre-proof status. Crossref currently deposits **volume 352**, October 2026 publication, and creation on 2026-09-05. These are different date fields; do not infer a precise October publication day from license-start dates, or treat an indexed pre-proof label as confirmed current production status. Semantic Scholar independently matches the DOI/title/year, but its CLOSED/no-PDF field does not prove that no lawful author copy exists.

Pons is a preprint, not a verified accepted journal paper. AlphaEdit/AlphaEdit+ have official proceedings records. These are controlled computer-science benchmark studies, roughly Level III in the skill's broad evidence rubric, not clinical RCT evidence. KBS methodology quality is ungraded pending full text. No exhaustive indexing, author-conflict, or retraction audit was performed; the indexed KBS declaration reports no competing financial/personal interests.

## Pons: exact threat and protocol

Evidence locations use **1-based PDF page numbers** because the paper's own footer numbering is inconsistent: §3.1–3.3 and Eqs.(1)–(3), PDF **p.5**; §4 and rank Eq.(4), **p.6**; §4.1 and CF Eq.(6), **p.7**; §6.1, **pp.10–11**.

Eq.(2) attributes rank failures to old-entity representation drift; Eq.(3) allows a newly introduced entity to outrank the old answer without requiring old embeddings to change. Eq.(4) computes filtered rank over candidate entities. §4 corrects evaluation of an old test snapshot j after training snapshot i by using **E_i**, not E_j. Eq.(6) adjusts forgetting for update count, baseline performance and test-set size. §6.1 uses eight FB15k-237-derived continual benchmarks, TransE at dimension 200, and final test-set-size-weighted MRR/Hits@1,3,10. These are continual graph-growth experiments, not independent one-column edits.

**Implication (our inference):** a claim that preserving query representations or old positive scores suffices to preserve predictions is already contradicted by this source. Our fixed-entity edit does not reproduce their new-entity-growth condition; however, moving an existing candidate above a protected answer causes the same rank-level competition. That motivates checking competitor scores, but does not make the general observation new. Pons does not establish that our particular finite inequality system or fallback policy has appeared before.

### Author-source inspection and reproduction caveats

[RevisitingCKGE](https://github.com/gerardponsrecasens/RevisitingCKGE) pinned at **93d7cb4a54e0336816a5b3d308ebcd90d9fe864a**. The no-checkout source clone and selected byte snapshots are in threat_review/pons-source and pons-files. The repository contains benchmark trees ENTITY, RELATION, FACT, HYBRID, graph_equal, graph_higher, graph_lower and **PS-CKGE-3_1_1**, plus seven bundled method implementations.

Checked LKGE paths at that commit:

- [data_loader.py](https://github.com/gerardponsrecasens/RevisitingCKGE/blob/93d7cb4a54e0336816a5b3d308ebcd90d9fe864a/experiments/LKGE/src/data_load/data_loader.py): lines 46–54 expand label-vector dimensions with full_entities; lines 83–86 attach test filters from the test snapshot's sr2o_all.
- [BaseModel.py](https://github.com/gerardponsrecasens/RevisitingCKGE/blob/93d7cb4a54e0336816a5b3d308ebcd90d9fe864a/experiments/LKGE/src/model/BaseModel.py): lines 159–164 expand test candidates.
- [main.py](https://github.com/gerardponsrecasens/RevisitingCKGE/blob/93d7cb4a54e0336816a5b3d308ebcd90d9fe864a/experiments/LKGE/main.py): lines 331–342 set/reset full_entities; 419–447 aggregate corrected final results.
- **Split caveat:** [KnowledgeGraph.py](https://github.com/gerardponsrecasens/RevisitingCKGE/blob/93d7cb4a54e0336816a5b3d308ebcd90d9fe864a/experiments/LKGE/src/data_load/KnowledgeGraph.py), line **27**, loads valid_facts from **test.txt**. main.py 113–116 selects/saves the best model using validation MRR. Thus this pinned LKGE path selects checkpoints using the nominal test facts. This is a source-observed limitation; it does not establish that every reported paper number or every bundled method used this exact path.
- **Filter/tie caveat:** expanding candidates does not update the old test snapshot's filter dictionary. model_process.py 79–85 masks its supplied labels and ranks by double argsort, whereas paper Eq.(4) uses strict greater-than. Current-snapshot filters and optimistic ties cannot silently be assumed. For a controlled rerun, audit both independently and declare any corrections as protocol changes.
- **Launcher caveat:** LKGE's launcher uses PS-CKGE and default five snapshots, but the available PS dataset has three numbered snapshots and a different folder name. IncDE/FastKGE launchers explicitly specify three snapshots. Resolve path mapping before launching.
- **Execution caveat:** main.py 162–165 removes an existing configured checkpoint subtree, and 197–199 unconditionally uses CUDA. Use a disposable run copy and a unique checkpoint prefix for every parallel run. A CPU-only execution or concurrent reuse of the same run directory is not established.
- FastKGE's supplied launcher includes a learning-rate/rank grid. Do not present running that grid as one fixed, budget-matched baseline.

**Priority:** Pons-corrected evaluation is mandatory if a later experiment introduces graph growth. Start with a bounded native ENTITY run and a strong bundled CKGE comparator after split/filter audit. The ETT-CKGE ENTITY launch is documented in pons-files/experiments/ETT-CKGE/run_experiments.sh; this sidecar has inspected its launcher, not its entire numerical implementation. Such native runs test continual learning; they are not directly comparable to our one-column SafeSucc.

## KBS: established collision, unresolved mechanics

Crossref verifies title, authors, journal, volume and DOI. The indexed publisher **abstract and Introduction** describe a PLM backbone plus editing branch/hypernetwork, neighborhood/relation-prototype locality regularization, low-rank transformation and relation-conditioned gating. The **Experimental setup** excerpt identifies PyTorch on one RTX 4090; **efficacy-efficiency analysis** mentions E-FB15k-237/E-WN18RR and Succ@1/RK@3.

**Implication (our inference):** generic claims of first locality-aware, structure-aware, efficient or relation-aware KGE editing collide with this work. A frozen DistMult/ComplEx candidate-column QP has a different optimization object, but model-family scope alone is not a novelty contribution. Its reported Succ@1 cannot be compared with our Top10 SafeSucc. Neither exact rank preservation nor absence of such preservation is established here.

**Missing before author reproduction:** complete Methods equations and their numbering; exact LM backbone/checkpoints; edit generation and split files; neighborhood construction/filtering; RK@3 definition; trained hypernetwork/gate weights; selected hyperparameters; author-linked source and instructions. Full publisher access returned **403**; Elsevier's deposited XML endpoint returned **429**; DOI resolution returned only a linking page. OpenAlex was rate-limited; an Unpaywall request was rejected and supplies no availability evidence. Exact-title/DOI/code searches exposed no authenticated author repository in the returned evidence. This is **incomplete verification**, not proof of code absence. No page/equation identifiers are invented. Preserve the citation as a verified paper with explicitly unresolved mechanics, not as a fully read strong baseline.

## AlphaEdit / AlphaEdit+: mechanistic collision

**AlphaEdit primary locations:** [v4 §3.1–3.3](https://arxiv.org/html/2410.02355v4#S3.SS1), Eqs.(7)–(14); §4.1; Appendix A.2–A.3. Eq.(10) preserves W K0 by restricting an update to Delta P with P K0=0. Eqs.(12),(14) fit new key/value targets with regularization and prior-edit penalties. The paper uses a thresholded covariance eigenspace in practice; its exact algebraic zero-change statement does not automatically become an exact whole-model or rank guarantee.

**AlphaEdit+ primary locations:** §2.1 Eqs.(1),(2), **p.14810**; conflict Eq.(3), pp.14810–14811; §3.1 Eq.(7), **p.14812**; §3.2 Eqs.(8),(9)/Algorithm 1, **p.14813**; §4.1, pp.14814–14815; Appendix B.1/Table 5, **p.14823**. Eq.(7) replaces P by P+Ptilde, adds conflict-dependent prior-edit weights and smooths large-residual targets. Eq.(8) searches preservation-covariance directions; Eq.(9) evaluates smoothing candidates. Appendix B.1 describes 3,500 AlphaSet examples and 550 validation instances. These are LLM editing protocols with efficacy/paraphrase/specificity scores.

### Author code and feasibility

[AlphaEdit](https://github.com/jianghoucheng/AlphaEdit), commit **b84624f44dfe8fc6cd9e41df916c44124a0c46dc**, MIT license. AlphaEdit_main.py 130–132 implements the projected linear solve; experiments/evaluate.py 426–443 constructs the thresholded covariance projector. Llama3-8B.json uses nullspace_threshold=0.02 and L2=10. The README lists one **48 GB A40** and dependencies; it supplies a native Llama3/CounterFact command. Its covariance-download link text and actual href differ, so verify the downloaded matrix's provenance/shape rather than trusting the visible label.

[AlphaEdit_plus](https://github.com/zjh-vinky/AlphaEdit_plus), commit **b4fe675ad638fe9fef5046ea3300536d09bd6ddf**. No top-level LICENSE was found in the pinned tree; do not infer a reuse license from the base repository. It has author commands and GPT2-XL/GPT-J/Llama3 configurations; its README/paper specify an **80 GB A100**. This does not prove that a smaller model cannot fit a 3090, but a faithful 3090 run has not been established here.

Source details in AlphaEditPlus_main.py: 155–208 adds covariance directions in batches of ten; 215–241 constructs smoothing residuals; 403–423 solves the weighted projected system; 458–482 computes conflict weights. Lines **268–306 select among smoothing candidates by evaluating the current record_chunks**, and experiments/evaluate.py 283–293 passes the current edit chunk. This source path does not implement the paper's separately constructed validation set as described. Preserve the discrepancy; a disjoint-validation repair is a declared adaptation. The bundled AlphaSet source files contain **100 K0, 100 KI and 200 KP entries**, not the complete 3,500/550-instance paper protocol.

These are source-backed execution entry points, not successful reproductions. Native code availability is substantially stronger than a search-result claim, but neither repository has been runtime-tested by this sidecar.

## Exact implication for our QP/fallback

Independent implementation references: project src/editing_policy.py lines 19–45; src/rank_constraints.py 12–33; src/rank_constraints_v2.py 12–67; src/prediction_protection_v2.py 10–46; src/true_ablation_policy.py 5–7. The inspected local source hashes are in threat_review/local-method-source-hashes.json. This inspection establishes code behavior, not current remote experiment status.

For a frozen query q_u and only one changed candidate tail t, the score change is **q_u^T delta**. The implementation minimizes **0.5 ||delta||²** subject to **A delta >= b**:

- Promotion requires q_target^T delta at least the score gap to Top10.
- TRUE rows prevent the edited candidate from crossing a protected rank-10 gold answer, or prevent the edited gold tail from falling below its competing boundary.
- Prediction rows use stored pre-edit Top10 membership/ties on TRAIN query keys and both upper/lower boundary inequalities. The requested query key is excluded.
- TRUE-only removes the prediction rows and leaves the rest of the policy intact.
- QP adoption requires solver success, original-unit primal checks, numerical KKT evidence, full-system recheck, actual target rank, covered TRAIN locality and norm no greater than paired Rank plus tolerance. Otherwise the returned update is **Rank(.125)**.

**Why these threats matter:** AlphaEdit preserves linear responses with equalities; our guards allow nonzero changes up to rank boundaries. The distinction is task-level inequalities, not a new discovery of protected subspaces or quadratic editing. AlphaEdit+ already relaxes preservation under conflict; fallback is a different operational policy but does not establish priority for conflict handling. Pons shows why competing entities must be evaluated, and KBS supplies structural-locality/efficiency prior work.

For a shallow response-space adaptation, identify K0 with protected query columns and Delta with delta^T. Then AlphaEdit's equality requirement becomes **q_u^T delta=0**. The promotion gap and prior-preservation constraints may conflict. Comparing that equality baseline with our inequality policy would isolate the consequence of rank slack, but is **our shallow-KGE adaptation**, not the authors' LLM protocol.

**Critical limits:** constraints certify only the declared fixed-column finite set. If edited entities also serve as query heads, the query itself changes; DistMult and ComplEx have additional head/tail effects, and self-involving triples can introduce nonlinear terms. Neither the QP certificate nor an AlphaEdit response-space adaptation guarantees shared-entity or unseen-query safety. The Rank fallback can damage facts or fail edits; zero damage for accepted QP candidates is not zero damage for the hybrid method. Matching update norm is not matching runtime. Previously exposed VALID/TEST outcomes cannot be relabeled as blind evaluation.

## Runnable author-source options for the execution owner

The clones in threat_review are intentionally **no-checkout**; selected source bytes are separately materialized. Commands below are documented options and **were not executed**. The execution owner must choose an isolated environment/run root, inspect free resources and preserve existing checkpoints.

1. **Pons native corrected-CKGE path:** materialize pinned experiments/LKGE and the desired benchmark; place/symlink ENTITY under its data/ENTITY directory. In a fresh run copy with a unique checkpoint prefix, the author launcher provides:

~~~bash
python -u main.py -dataset ENTITY -gpu 0 -lifelong_name LKGE \
  -learning_rate 0.0001 -batch_size 2048 -seed 11 -note 11
~~~

   Dependencies include torch, numpy and prettytable; CUDA is assumed by source. Author seeds are 11,22,33,44,55. Use the native split behavior only when explicitly labeling a source replay; a controlled comparison must correct test-backed validation and document filter/tie conventions. Do not launch the full sweep automatically. PS-CKGE additionally needs the three-snapshot/path mapping resolved.

2. **AlphaEdit native example:** in a materialized isolated clone at the pinned revision, with model access, compatible dependencies and covariance statistics:

~~~bash
python3 -m experiments.evaluate --alg_name=AlphaEdit \
  --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname=Llama3-8B.json --ds_name=mcf \
  --dataset_size_limit=2000 --num_edits=100 --downstream_eval_steps=5
~~~

   This is the README's native example. GPT2-XL configurations exist, but changing model/samples/hardware is a separate scoped run. The native evaluation is LLM CounterFact, not KGE ranking.

3. **AlphaEdit+ native example:** the README expects the GPT2-XL model at ../models/gpt2-xl and supports:

~~~bash
python3 -m experiments.evaluate --alg_name=AlphaEdit_plus \
  --model_name=../models/gpt2-xl --hparams_fname=gpt2-xl.json \
  --ds_name=zsre --dataset_size_limit=200 --num_edits=100 \
  --downstream_eval_steps=1
~~~

   Audit the current-chunk candidate-selection issue, incomplete AlphaSet and hardware fit before treating a run as a clean paper reproduction. Do not silently tune projector thresholds, replace validation or scale smoothing to seek a favorable result.

4. **KBS:** no author-source command can responsibly be supplied from the evidence acquired. Retrieve the complete paper and its actual code/data/checkpoint declarations before implementing an approximation. No baseline placeholder should be described as the authors' method.

**Recommended order:** first adopt the candidate/score/filter audit relevant to the chosen task; next run a declared shallow equality/projection comparator beside TRUE/Prediction under identical checkpoints/cohort/information; separately reproduce selected native author protocols when feasible. Keep these result tables separate. KBS remains an unresolved high-priority source request. None of this licenses a “first”, “SOTA”, or submission-GO claim.

## Provenance and verification limits

threat_review/retrieval-log.json records URLs, resolutions, HTTP failures, timestamps and hashes. provenance-manifest.json pins repositories and records source-file/PDF hashes. kbs-publisher-indexed-excerpt.txt preserves the publisher-indexed receipt with an explicit partial-access label; it is not a downloaded full article. HTML text extraction and static Python AST parsing are source-reading aids only; no method import, training, model download or empirical baseline run occurred. The report was assisted by AI research tools and remote source inspection.

Metadata cross-checks used independent Crossref/Semantic Scholar and official proceedings where available. Source equations establish mechanistic overlap; they do not settle all algorithm-priority questions. Bounded unsuccessful access/search attempts never establish nonexistence. The full KBS verification and faithful runnable 3090 demonstrations remain missing.


## Handoff update: KGEditor resources supplied by main

The user reports that main has located these Hugging Face resources: [model zjunlp/KGEditor](https://huggingface.co/zjunlp/KGEditor), [dataset ChancesYuan/KGEditor](https://huggingface.co/datasets/ChancesYuan/KGEditor), and [space zjunlp/KGEditor](https://huggingface.co/spaces/zjunlp/KGEditor). These pointers are user-supplied and were not independently inspected by this sidecar; they are not evidence about KBS code availability or equivalence. Main owns their downloads and environment work.

When downloaded source is ready, a bounded follow-up can audit paper/repository attribution, entity/relation IDs and inverse-relation mapping, backbone/checkpoint compatibility, edit-versus-preservation splits, candidate/filter/tie conventions, Success/Retention definitions, and whether validation or evaluation examples are used for editing or checkpoint selection. Native protocol and shallow-KGE adaptation must remain distinct. No KGEditor training or download was launched by this role.
