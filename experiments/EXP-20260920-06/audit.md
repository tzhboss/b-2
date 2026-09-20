# Experiment Audit — EXP-20260920-06

## Identity

- Experiment ID: EXP-20260920-06
- Preregistered commit: ee22d9627dd2bdf133fe84a1ae1b178d8c4140f8
- Protocol: configs/protocols/reference_target_diagnostic_v1.yaml

## Completeness

- Metric rows: 720 / 720 expected.
- Reference-error summary rows: 12 / 12 expected.
- Paired-delta rows: 36 / 36 expected.
- Metric NaNs: zero.
- Exact EXP-05 speaker-fold and enrollment-generation functions were reused.

## Registered Criteria

### Marginal-target convergence

Supported in all three datasets. Mean absolute error to the full speaker marginal median falls from K=1 to K=10 by:
- ESD: 33.9%.
- MEAD: 61.9%.
- RAVDESS: 72.4%.

### Target-identity explanation

Supported in all three datasets. At K=10, the random label-free enrollment median is closer to the marginal reference than the neutral reference:
- ESD: 1.374 st to marginal vs 3.107 st to neutral.
- MEAD: 0.721 vs 2.001 st.
- RAVDESS: 1.082 vs 4.104 st.

### Functional-equivalence criterion

The strict within-0.01 criterion is met only on RAVDESS. It fails on ESD and MEAD because the marginal-oracle reference is not worse but significantly better than the neutral-oracle reference:
- ESD marginal minus neutral oracle: +0.0203 macro-F1, CI [0.0029, 0.0365].
- MEAD: +0.0140 [0.0061, 0.0222].
- RAVDESS: +0.0039, CI crosses zero.

## Audit Verdict

- Validity: valid.
- Decision: mixed.
- K-shot convergence to an unlabeled marginal speaker center: supported.
- Neutral-reference uniqueness: contradicted by the observed marginal-oracle advantage on ESD and MEAD.
- Strict functional-equivalence criterion: not met because marginal reference can outperform neutral reference.
