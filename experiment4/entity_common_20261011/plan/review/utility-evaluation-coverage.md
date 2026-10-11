# Fixed VALID cohort coverage; wording correction only

The original producer selects1024 old and1024 new directed VALID queries
with seed124. The independent proposal/shared-graph utility auditors rebuild
those same cohorts and score every11258 candidate at every selected query,
plus the old7505 pool for the originally successful old-query denominator.
This is complete scoring of the fixed cohort, not evaluation of every query
in each dataset VALID split. The literal legacy status
`PASS_FULL_COMMON_VALID_METRICS` refers to that selected-cohort reconstruction.

Evidence is the query selection in `scripts/audit_proposal_utility.py` and
`scripts/audit_shared_graph_utility.py`, the locked producers' respective
validation_queries_per_split/validation_seed values, and their saved old/new
query hashes.038 delegates the unchanged shared-graph auditor and additionally
checks all original top10 TRAIN truths at the fixed exempted keys.

Each shared-graph utility audit therefore reconstructs2048 selected query
identities and5120 ranks: source and endpoint old/new cohorts plus old-pool
old-query ranks. It does not independently retrain source SGD. These cohorts
are exposed development evidence; seedwise and directionwise gates remain
unchanged. Do not expand or replace their query selection after results.

The common comparison table separately evaluates all74428 old and12406 new
directed TEST queries against11258 candidates. Common inference audits check
all query identities and metric aggregation plus64 complete candidate scoring
rows, while graph guard audits check every compiled terminal TRAIN query
group on CPU and all3753 saved patch/state/budget records. They are distinct
forms of coverage; neither proves independently rescoring all TEST queries
on CPU or retraining every baseline.

The current generated report previously said full/all VALID in two places.
Correct the prose in the builder/table descriptions while retaining old
commits, frozen producer/auditor sources, configs, status labels, metrics,
query identities, outcomes and scientific gates. No experiment is rerun and
no negative decision is reversed by this correction.
