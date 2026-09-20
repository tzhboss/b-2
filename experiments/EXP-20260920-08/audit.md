# Experiment Audit — EXP-20260920-08

## Completeness

- Fold-level metric rows: 300 / 300 expected.
- Summary rows: 20 / 20 expected.
- Paired-delta rows: 20 / 20 expected.
- Inventory rows: 5 / 5 expected.
- Metric NaNs: zero.
- Speaker baseline is constant within speaker to floating-point precision in all five corpora.

## MSP registered mechanism criteria

Speaker-balanced Macro-F1:
- Relative: 0.2054.
- Absolute: 0.2290.
- Baseline-only: 0.1762.
- Relative+Baseline: 0.2385.

Paired deltas:
- Relative+Baseline minus Relative: +0.0332, 95% CI [0.0284, 0.0377].
- Absolute minus Relative: +0.0236 [0.0204, 0.0267].
- Relative+Baseline minus Absolute: +0.0095 [0.0052, 0.0135].

The preregistered baseline-utility mechanism criterion is satisfied.
The preregistered Baseline-only > 0.12 support criterion is also satisfied.

## Cross-corpus context

Adding baseline to Relative:
- ESD: +0.0165, CI above zero.
- MEAD: +0.0046, CI crosses zero.
- RAVDESS: +0.0176, CI above zero.
- MELD: -0.0172, CI below zero.
- MSP: +0.0332, CI above zero.

The baseline contribution is therefore also corpus-dependent rather than universally beneficial.

## Verdict

- Validity: valid.
- Decision: pass for the registered MSP baseline-utility mechanism.
