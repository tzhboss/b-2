# Experiment Audit — EXP-20260920-14

## Completeness

- Fold-level metric rows: 1,350 / 1,350 expected.
- Summary rows: 90 / 90 expected.
- Delta rows: 60 / 60 expected.
- Family-comparison rows: 15 / 15 expected.
- Inventory rows: 15 / 15 expected.
- Metric NaNs: zero.
- Duplicate evaluation keys: zero.
- Same fixed seeds/folds and matched rows were used for both classifier families.

## Registered Criteria

Relative-minus-Absolute effects:
- Spearman(Logistic, HGB) across 15 corpus-by-attribute cells: 0.5643.
- Required for acceptance: >=0.60.
- Sign agreement among |Logistic delta| >=0.01 cells: 0.70 (7/10).
- Required for acceptance: >=0.80.

Relative+Baseline-minus-Relative:
- Spearman(Logistic, HGB): 0.20.
- Sign agreement among |Logistic delta| >=0.01 cells: 0.667 (6/9).
- Required for acceptance: >=0.70.

The broad rejection criterion is not met because Relative-minus-Absolute correlation is above 0.30
and direction agreement is above 0.60.

## Important Direction Flips

Large/meaningful Logistic effects that reverse under HGB include:
- MSP pitch: Logistic -0.0236, HGB +0.0091.
- MSP rate: Logistic -0.0773, HGB +0.0048.
- MELD rate: Logistic -0.0117, HGB +0.0044.

Baseline-addition flips include:
- MELD pitch.
- MSP loudness.
- MEAD rate.

## Verdict

- Validity: valid.
- Decision: inconclusive.
- Broad classifier-family robustness is not supported at the preregistered thresholds.
- Reference-frame utility is itself partly classifier-dependent.
