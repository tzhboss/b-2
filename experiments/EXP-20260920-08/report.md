# Experiment Report — EXP-20260920-08

## Main result

MSP's absolute-pitch advantage is substantially explained by useful information in the stable
speaker pitch baseline that is removed by speaker-relative normalization.

On speaker-disjoint MSP:
- Relative speaker-balanced Macro-F1: 0.2054.
- Absolute: 0.2290.
- Relative + explicit speaker baseline: 0.2385.
- Baseline-only: 0.1762.

Adding baseline back to Relative improves speaker-balanced Macro-F1 by +0.0332 with a paired
95% confidence interval [0.0284, 0.0377]. The decomposed two-feature representation also exceeds
raw Absolute pitch by +0.0095.

## Interpretation

Relative pitch itself remains emotion-informative on MSP, but normalization discards a stable
speaker-level component that is also predictive of emotion labels in this corpus. Absolute pitch
therefore wins over Relative not because within-speaker deviation is useless, but because the
removed baseline is functionally useful.

The sign of baseline utility remains corpus-dependent: it helps on ESD, RAVDESS, and MSP,
is weak on MEAD, and hurts on MELD.

## Decision

pass for the registered MSP mechanism hypothesis.
