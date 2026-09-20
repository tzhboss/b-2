# Experiment Audit — EXP-20260920-05

## Identity

- Experiment ID: EXP-20260920-05
- Runtime preregistration commit: 9995830158f46245d3f7e9c915763d74cc5ac750
- Protocol: configs/protocols/unseen_speaker_kshot_pitch_v1.yaml
- Experiment config: configs/experiments/EXP-20260920-05.yaml

## Evaluation Audit

- Fold-level metric rows: 900 / 900 expected.
- Baseline speaker-evaluation rows: 984 / 984 expected.
- Paired-delta rows: 60 / 60 expected.
- Seeds: 3.
- Speaker-disjoint folds: 5.
- K values: 1, 2, 5, 10.
- Representations: Absolute, Oracle-relative, K-shot-relative, Oracle-hybrid, K-shot-hybrid.
- Metric NaNs: zero.
- Maximum train/test speaker overlap: zero.
- Maximum enrollment/evaluation row overlap: zero.
- Full emotion-class coverage retained.
- Enrollment selection is uniform random within speaker and does not use emotion labels.
- The precomputed oracle pitch baseline is constant within each speaker up to floating-point precision.

## Registered Criterion Evaluation

### Baseline-estimation criterion

Not supported. K=10 does not move monotonically toward the neutral-derived oracle baseline:
- ESD MAE: 2.129 st at K=1 versus 3.107 st at K=10.
- MEAD: 2.862 -> 2.001 st.
- RAVDESS: 5.356 -> 4.104 st.

Only MEAD reaches the preregistered >=25% reduction; RAVDESS improves by about 23%, and ESD becomes worse.

### Downstream-utility criterion

Supported in all three datasets at K=10:
- ESD K-shot Relative minus Absolute: +0.1065 macro-F1, 95% CI [0.0684, 0.1432].
- MEAD: +0.0980 [0.0907, 0.1061].
- RAVDESS: +0.0744 [0.0530, 0.0974].

### Practical-recovery criterion

Supported in all three datasets. K=10 K-shot Relative is within 0.02 macro-F1 of Oracle-relative:
- ESD: -0.0021.
- MEAD: +0.0077.
- RAVDESS: +0.0098.

All three paired CIs include zero for K-shot Relative versus Oracle-relative.

## Audit Verdict

- Validity: valid.
- Decision: mixed.
- Baseline-convergence-to-neutral hypothesis: not supported.
- Unseen-speaker K-shot downstream utility: strongly supported.
- Practical recovery of oracle-level downstream performance by K=10: supported.
