# Experiment Audit — EXP-20260920-01

## Identity

- Experiment ID: EXP-20260920-01
- Runtime commit: 4e7ea52eeab59a171a8b8cf297d4f109c1cd3daa
- Branch: main
- Protocol: configs/protocols/prosody_reference_frame_v1.yaml
- Experiment config: configs/experiments/EXP-20260920-01.yaml

## Data and Environment

- Data version/hash: final_dataset_enriched.parquet, SHA256 40a537663a111de36d0ba26f4ba43368592004b8c2290e29e4f4da05600774b4
- Environment/runtime: /data/lc/audio_feature_labeling/.venv, Python 3.12.13
- Seed(s): 20260920, 20260921, 20260922

## Outputs

- Expected outputs: all protocol-required raw artifact files.
- Actual outputs: all expected files plus row_hash_checks.csv.
- Missing outputs: none.
- Checkpoint integrity: not applicable; fixed logistic-regression probes.
- Evaluation completeness: 1080 metric rows; all 3 seeds, 5 folds, 3 representations, 3 datasets, 2 tasks, 4 attribute sets completed.

## Protocol Compliance

- Row identity: identical across Absolute/Relative/Hybrid for every matched comparison.
- Train-only scaler: enforced by sklearn Pipeline fitted separately on each training fold.
- Protocol violations: RAVDESS-speech emotion fold 4 lacks the neutral class for all three seeds. The implementation computed macro-F1 without an explicit global label set, so that fold used a different class universe from folds 0–3.
- Deviations and rationale: none intentionally introduced. The class-coverage/metric issue was discovered only after execution from sklearn warnings and a post-run coverage audit.

## Audit Verdict

- Validity: invalid
- Evidence supporting verdict: 12 RAVDESS emotion dataset/attribute/seed fold-coverage cases omit neutral in fold 4; macro-F1 is therefore not strictly comparable across all required folds.
- Follow-up required: rerun under a superseding protocol with four folds and an explicit assertion that every train/test fold contains the full dataset-task class set; keep all model and representation choices unchanged.
