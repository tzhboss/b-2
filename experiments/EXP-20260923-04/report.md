# Experiment Report — EXP-20260923-04

## Audit repair

EXP-20260921-06 used a defective integer seed derivation that caused nominal permutation reps to
repeat the same derangement. This experiment preserves the historical result and reruns the MSP
control with SHA-256-derived seeds and auditable train/test mapping hashes.

All 30 seed×fold×task randomization cells contain exactly **20 permutation reps and 20 distinct
mapping hashes**. The corrected randomizations have nonzero outcome variance.

## Corrected descriptive result

Mean true-center Hybrid minus mean genuinely permuted-center Hybrid across the 15 split cells:

- Raw Arousal: **+0.29545**; split-cell range [+0.24120,+0.32684].
- Raw Dominance: **+0.23654**; [+0.19618,+0.26841].
- Within-speaker residual Arousal: **+0.00140**.
- Within-speaker residual Dominance: **+0.00101**.

Thus repairing the seed bug leaves the substantive point estimate essentially unchanged: correct
speaker-center pairing carries large predictive value when the raw MSP target retains
between-speaker structure, while the identity-specific advantage nearly vanishes after target
centering.

## Statistical boundary

The ranges/std values here describe split/randomization variability. They are not confidence
intervals over 15 independent samples; seed×fold cells are not treated as independent speakers.
The old fold-pair bootstrap claim is retired. A separate same-budget [x;mu] versus [x;pi(mu)]
control is registered as EXP-20260923-07 to address the remaining mechanism-identification issue.

## Decision

pass — randomization bug repaired; core point estimate survives genuine independent permutations.
