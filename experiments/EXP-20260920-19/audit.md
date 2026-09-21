# Experiment Audit — EXP-20260920-19

## Completeness and leakage

- Fold-level metric rows: 1,260 / 1,260 expected.
- Paired-delta rows: 72 / 72 expected.
- Audit rows: 60 / 60 expected.
- Metric NaNs: zero.
- Train/test speaker overlap: zero.
- Enrollment/evaluation overlap: zero.
- Enrollment sampling uses no VAD labels.

## K-shot acoustic-center convergence

K=1 to K=10 mean absolute error reduction to the full marginal speaker center:
- Pitch: 67.8%.
- Loudness: 68.7%.
- Log-rate: 70.6%.

Thus the registered center-convergence criterion is supported.

At K=10, K-shot estimates are much closer to the marginal speaker center than to the
neutral-derived reference:
- Pitch: 0.680 vs 1.557 semitone-equivalent units.
- Loudness: 0.696 vs 1.789 LU/dB.
- Log-rate: 0.055 vs 0.076.

## K=10 raw-VAD Hybrid utility

K-shot Hybrid minus Absolute CCC:
- Arousal: +0.01944, 95% CI [0.01651, 0.02221].
- Dominance: +0.00923, 95% CI [0.00739, 0.01091].
- Valence: +0.01377, 95% CI [0.01290, 0.01476].

The preregistered deployment criterion required at least +0.015 Arousal and +0.010 Dominance.
Arousal passes; Dominance falls short by 0.00077, so the joint criterion is not met.

## Oracle recovery

K=10 Hybrid minus marginal-oracle Hybrid:
- Arousal: -0.00529.
- Dominance: -0.00050.
- Valence: -0.00280.

Arousal and Dominance are both within the preregistered 0.02 oracle-recovery tolerance.

## Verdict

- Validity: valid.
- Decision: mixed.
- Center convergence: supported.
- Deployment Hybrid joint threshold: not fully supported because Dominance narrowly misses the fixed +0.010 criterion.
- Practical recovery of marginal-oracle Hybrid performance: supported.
