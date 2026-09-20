# Experiment Audit — EXP-20260920-13

## Result

Nested confidence-aware routing does not improve over the naive EXP-12 stats-only router.

Aggregate mean F1:
- Training-best fixed: 0.2149.
- Naive stats router: 0.2246.
- Nested router: 0.2177.
- Oracle: 0.2306.

Nested gain over fixed is only +0.0028, below the preregistered +0.005 threshold.
Nested oracle regret is 0.0129, about 82.3% of fixed-policy regret and above the required 75%.

Per-corpus nested gain over training-best fixed:
- ESD: 0.0000.
- MEAD: -0.00075.
- MELD: +0.00503.
- MSP: +0.01101.
- RAVDESS: -0.00141.

Only two corpora improve, so the robustness criterion is not met.

Selected thresholds are training-only and vary by outer corpus:
ESD 0.8, MEAD 0.5, MELD 0.7, MSP 0.9, RAVDESS 0.6.

## Verdict

- Validity: valid.
- Decision: reject for the registered confidence-aware routing hypothesis.
- The naive stats-only router remains the better practical policy in this dataset.
- No additional threshold tuning is justified from these same corpora.
