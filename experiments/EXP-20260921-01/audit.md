# Experiment Audit — EXP-20260921-01

## Runtime integrity

- Source and runtime script SHA256 match exactly.
- Source and runtime config SHA256 match exactly.
- Runtime executed from immutable /tmp copies.
- Experiment metadata reports the expected experiment ID, K sequence, and fixed-cohort rule.

## Completeness

- Fixed cohort: 884 speakers, 169,429 rows, minimum 51 rows per speaker.
- Fold-level metrics: 810 / 810 expected.
- Summary rows: 54 / 54 expected.
- K-delta rows: 9 / 9 expected.
- Metric NaNs: zero.

## Acoustic-center sample efficiency

Marginal-center MAE:

Pitch:
- K1 2.3839
- K10 0.8256
- K20 0.5648
- K50 0.2641

Loudness:
- K1 2.5068
- K10 0.8077
- K20 0.6152
- K50 0.3078

Log-rate:
- K1 0.1826
- K10 0.0635
- K20 0.0432
- K50 0.0198

For all three dimensions, the absolute K10-to-K50 MAE improvement is less than half the
K1-to-K10 improvement, satisfying the preregistered center-saturation criterion.

## Downstream saturation

K-shot Hybrid CCC:
- Arousal: K10 0.4915, K20 0.4962, K50 0.5018.
- Dominance: K10 0.3827, K20 0.3846, K50 0.3922.

K50 minus K10:
- Arousal: +0.01028.
- Dominance: +0.00946.

The preregistered <0.01 condition is satisfied for Dominance but narrowly missed for Arousal by
0.00028 CCC. Therefore the K=10 practical-plateau criterion is not fully supported.

K50 minus K20:
- Arousal: +0.00564.
- Dominance: +0.00761.

These K20 results are descriptive because K20 was not the preregistered practical-plateau target.

## Verdict

- Validity: valid.
- Decision: mixed.
- Acoustic-center diminishing returns: supported.
- K=10 downstream plateau: narrowly not supported.
- Descriptively, K=20 is within 0.01 CCC of K=50 for both Arousal and Dominance.
