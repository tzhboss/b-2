# Experiment Report — EXP-20260923-07

## Question

When raw utterance features x are retained in every condition, does appending the correctly paired
speaker center help more than appending an equally sized wrong-speaker center?

Conditions:
- raw: x
- true-center: [x; mu_s]
- permuted-center: [x; pi(mu_s)]

All 30 seed×fold×task permutation cells contain 20/20 unique mapping hashes.

## Result

Raw absolute targets:
- Arousal: raw 0.46020; wrong-center 0.45996; true-center 0.48579.
  True minus raw **+0.02558**; true minus wrong-center **+0.02583**.
- Dominance: raw 0.37216; wrong-center 0.37198; true-center 0.38159.
  True minus raw **+0.00942**; true minus wrong-center **+0.00961**.

Thus a correctly paired center adds predictive information beyond raw x, while an unrelated center
with the same dimensional budget contributes essentially nothing.

However, the preregistered expectation that this identity-specific advantage would shrink on
within-speaker residual targets was false:
- Residual Arousal: raw 0.13893; wrong-center 0.13908; true-center 0.33553.
  True minus raw **+0.19660**.
- Residual Dominance: raw 0.09296; wrong-center 0.09307; true-center 0.23709.
  True minus raw **+0.14412**.

## Interpretation

This result clarifies that a correct speaker center has two conceptually different roles.

1. **Calibration role:** on raw absolute targets, mu_s carries speaker-level information that helps
   locate the speaker on the population scale.
2. **Reference role:** when raw x is retained, the correct mu_s lets the readout reconstruct
   speaker-relative deviation x-mu_s. That is highly useful for within-speaker residual targets.

This explains why EXP-20260923-04 found almost no residual benefit from appending the correct versus
wrong center *after the correct Relative features were already supplied*: the reference operation had
already been performed there. In the present raw-x control, the correct center is needed to perform
that operation.

Therefore the stronger statement "speaker center is useful only when the target contains
between-speaker structure" is rejected. The safer mechanism statement is:
**the usefulness of a speaker reference depends on how it is used—both for population-scale
calibration and for computing within-speaker deviation.**

## Decision

mixed but highly informative; preregistered raw-target hypothesis passes, residual-target hypothesis
is rejected and the mechanism interpretation is revised.
