# Experiment Audit — EXP-20260920-11

## Completeness

- Effect cells: 75 / 75 unique corpus × attribute × emotion combinations.
- Resolved-sign cells: 43.
- LOCO predictions: 225 = 75 cells × 3 predictor families.
- LOAO predictions: 225.
- Every held-out cell receives exactly one prediction from labels-only, stats-only, and combined models.
- Features are computed from raw acoustic/label statistics and do not use downstream model predictions.

## Leave-One-Corpus-Out

Pooled results:

| Family | MAE | Spearman r | All-cell sign acc. | Resolved sign acc. | Resolved balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| labels-only | 0.0500 | 0.060 | 0.573 | 0.581 | 0.520 |
| stats-only | 0.0460 | 0.550 | 0.760 | 0.791 | 0.817 |
| combined | 0.0476 | 0.479 | 0.707 | 0.721 | 0.661 |

The preregistered combined-model direction criterion is satisfied: 0.721 resolved-sign accuracy,
more than 0.05 above labels-only. The preregistered regression criterion is not satisfied because
combined MAE is not <=0.85 times labels-only MAE, despite Spearman r >0.45.

Stats-only is the strongest model family descriptively and clearly outperforms labels-only.

Held-out-corpus resolved sign accuracy for stats-only:
- ESD: 0.857.
- MEAD: 0.917.
- MELD: 0.667.
- MSP: 0.714.
- RAVDESS: 0.714.

## Leave-One-Attribute-Out

Combined resolved sign accuracy:
- held-out loudness: 0.500.
- held-out pitch: 0.941.
- held-out rate: 0.643.

Only one held-out attribute reaches the preregistered >=0.65 criterion; therefore broad
attribute-general predictive structure is not supported.

## Feature Diagnostics

The strongest raw univariate association with the target is relative-minus-absolute separation
(Spearman r about 0.60). Baseline effect magnitude is negatively associated with Relative
preference, consistent with the information-redistribution mechanism.

## Verdict

- Validity: valid.
- Decision: mixed.
- Corpus-general direction prediction: supported.
- Accurate cross-corpus magnitude prediction: not supported by the preregistered threshold.
- Broad leave-one-attribute-out generalization: not supported.
- Descriptive strongest family: stats-only.
