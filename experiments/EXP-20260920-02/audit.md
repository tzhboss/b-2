# Experiment Audit — EXP-20260920-02

## Identity

- Experiment ID: EXP-20260920-02
- Runtime commit: e04eeda8fc6aff508a41f6b53ba11a391809b219
- Branch: main
- Protocol: configs/protocols/prosody_reference_frame_v2.yaml
- Experiment config: configs/experiments/EXP-20260920-02.yaml

## Data and Environment

- Data source: legacy derived snapshot final_dataset_enriched.parquet; this experiment does not claim to use the unfinished current English-final release.
- Data SHA256: 40a537663a111de36d0ba26f4ba43368592004b8c2290e29e4f4da05600774b4
- Loaded rows after dataset selection: 50,674 before task/feature filtering.
- Environment/runtime: /data/lc/audio_feature_labeling/.venv, Python 3.12.13, CPU logistic-regression probes.
- Seeds: 20260920, 20260921, 20260922.
- Folds: four deterministic within-speaker-label folds.

## Outputs

- Expected outputs: data inventory, fold metrics, summary, paired deltas, run metadata.
- Actual outputs: all expected outputs plus row_hash_checks.csv.
- Missing outputs: none.
- Evaluation completeness: 864 fold-level metric rows = 3 datasets × 2 tasks × 4 attribute sets × 3 representations × 3 seeds × 4 folds.
- Paired comparisons: 72 rows = 3 datasets × 2 tasks × 4 attribute sets × 3 pairwise representation comparisons.
- Metric NaNs: zero.

## Protocol Compliance

- Source identity: observed and hash-pinned in run_metadata.json.
- Row identity: zero failures; Absolute, Relative, and Hybrid use the same row-ID hash for every matched dataset/task/attribute/seed/fold comparison.
- Train-only scaling: StandardScaler is inside the sklearn pipeline fit separately on each training fold.
- Full-label coverage: independently reconstructed after execution; zero train/test class-coverage failures.
- Required seeds/folds: all present.
- Runtime worktree: clean at experiment start according to run_metadata.json.
- Protocol violations: none observed in EXP-20260920-02.
- Known limitation required by protocol: the split is intentionally reference-available / known-speaker. It is not evidence for unseen-speaker deployment.
- Source-version limitation: the experiment uses a pinned legacy enriched snapshot because it already contains aligned absolute and speaker-relative pitch/loudness/rate fields. Current per-dataset English final gates remain incomplete.

## Audit Verdict

- Validity: valid
- Evidence supporting verdict: all required outputs exist, row identity and class coverage checks pass, the pinned source hash is recorded, and all planned seeds/folds/conditions completed without runtime warnings.
- Follow-up required: test uncontrolled corpora and unseen-speaker/reference-estimation protocols before broader generalization claims.
