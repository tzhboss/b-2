# Experiment Audit — EXP-20260920-21

## Completeness and fixed-pool control

- Common cohort: 884 speakers, 169,429 utterances, minimum support 51.
- Reserved enrollment: exactly 50 utterances per speaker.
- Fold-level metric rows: 810 / 810 expected.
- Paired representation-delta rows: 54 / 54 expected.
- Matched K-to-K delta rows: 15 / 15 expected.
- Audit rows: 90 / 90 expected.
- Metric NaNs: zero.
- Speaker overlap: zero.
- Reserved-enrollment/downstream overlap: zero.
- Test-row hashes and train/test row counts are exactly identical across all K within every seed/fold.

## Center Convergence

Marginal-center MAE decreases monotonically at every K for pitch, loudness, and log-rate.

K=10 to K=50 MAE reduction:
- Pitch: 66.2%.
- Loudness: 65.7%.
- Log-rate: 69.5%.

Center-convergence criterion is supported.

## Fixed-Pool Hybrid Performance

Arousal CCC:
- K1: 0.4795.
- K2: 0.4827.
- K5: 0.4839.
- K10: 0.4871.
- K20: 0.4903.
- K50: 0.4946.

Dominance CCC:
- K1: 0.3793.
- K2: 0.3811.
- K5: 0.3818.
- K10: 0.3837.
- K20: 0.3839.
- K50: 0.3840.

At K=20, K-shot Hybrid minus marginal-oracle Hybrid:
- Arousal: -0.00427.
- Dominance: +0.00036.

K50 minus K20:
- Arousal: +0.00428, 95% CI [0.00211, 0.00633].
- Dominance: +0.00013, CI crosses zero.

Thus the preregistered practical-saturation criterion is supported.

## Diminishing-Returns Criterion

The strict criterion required |K50-K20| < |K10-K5| for both Arousal and Dominance.
- Dominance satisfies it.
- Arousal does not: +0.00428 vs +0.00326.

Therefore strict diminishing returns are not supported, even though the practical effect is small
and K20 is already within 0.005 CCC of the marginal oracle.

## Verdict

- Validity: valid.
- Decision: mixed.
- Pure center convergence: supported.
- Practical saturation by K=20: supported.
- Strict diminishing-returns shape: not supported.
