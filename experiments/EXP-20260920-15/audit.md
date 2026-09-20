# Experiment Audit — EXP-20260920-15

## Completeness

- Per-emotion fold rows: 6,750.
- Per-emotion summary rows: 450.
- Delta rows: 150.
- Classifier-interaction rows: 75 / 75 expected corpus × attribute × emotion cells.
- Metric NaNs: zero.

## Registered Criteria

Relative-minus-Absolute class-level effects:
- Spearman(Logistic, HGB): 0.2345.
- Acceptance threshold: >=0.50.
- Sign agreement among |Logistic delta| >=0.03 cells: 0.60 (18/30).
- Acceptance threshold: >=0.75.

Baseline-addition effects:
- Spearman(Logistic, HGB): 0.0584.
- Sign agreement among |Logistic baseline delta| >=0.03 cells: 0.609 (14/23).
- Acceptance threshold: >=0.70.

The strict rejection criterion is not met because sign agreement equals 0.60 rather than falling below 0.60.

## Classifier-Sensitive Cells

- 26 / 75 cells have |HGB RA - Logistic RA| >= 0.05.
- 27 / 75 cells have |HGB baseline-effect - Logistic baseline-effect| >= 0.05.

Largest Relative-minus-Absolute classifier interactions include:
- MSP rate neutral: Logistic -0.4343 vs HGB +0.0342.
- MSP pitch neutral: -0.3229 vs -0.0192.
- ESD pitch sad: +0.1369 vs -0.0871.
- ESD pitch angry: +0.0999 vs -0.1237.
- MEAD pitch happy: +0.1901 vs +0.0097.
- RAVDESS pitch sad: +0.0802 vs -0.0959.

## Verdict

- Validity: valid.
- Decision: inconclusive.
- Broad class-level classifier-family robustness is not supported.
- A substantial minority of reference-frame effects are strongly classifier-sensitive.
