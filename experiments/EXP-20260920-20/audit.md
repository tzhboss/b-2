# Experiment Audit — EXP-20260920-20

## Completeness

- Common cohort: 881 speakers, 168,237 utterances, minimum support 51.
- Fold-level metric rows: 1,890 / 1,890.
- Paired-delta rows: 108 / 108.
- Audit rows: 90 / 90.
- Metric NaNs: zero.
- Train/test speaker overlap: zero.
- Enrollment/evaluation overlap: zero.

## Center Convergence

Marginal-center MAE decreases monotonically at every K for all three attributes.

K=10 to K=50 MAE reduction:
- Pitch: 66.6%.
- Loudness: 67.1%.
- Log-rate: 68.4%.

Registered center-convergence criterion is supported.

## VAD Sample Efficiency

K=10 recovered fraction of marginal-oracle Hybrid gain over Absolute:
- Arousal: 62.2%.
- Dominance: 61.5%.

The preregistered >=70% K=10 efficiency criterion is not met.

Hybrid CCC:
- Arousal: K1 0.4857, K2 0.4875, K5 0.4923, K10 0.4968, K20 0.5041, K50 0.4971.
- Dominance: K1 0.3805, K2 0.3803, K5 0.3818, K10 0.3831, K20 0.3853, K50 0.3856.

At K=20, both Arousal and Dominance are within 0.002 CCC of marginal-oracle Hybrid.
However, Arousal K50 is 0.0069 lower than K20, exceeding the preregistered <0.005 saturation-change bound.

## Design Boundary

Because K enrollment utterances are excluded from downstream train/test rows, increasing K changes
both reference-estimation quality and downstream sample availability. The non-monotonic Arousal
curve therefore cannot be interpreted as a pure reference-estimation saturation effect.

A fixed-pool follow-up is required: reserve 50 enrollment utterances for every K, keep downstream
rows identical, and estimate the center from the first K nested enrollment utterances.

## Verdict

- Validity: valid.
- Decision: mixed.
- Acoustic-center convergence: supported.
- K=10 efficiency: not supported.
- Practical saturation: not established because of the K-dependent downstream-row confound.
