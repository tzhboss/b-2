# Experiment Report — EXP-20260920-06

## Main Result

The apparent failure of K-shot enrollment to approach the neutral speaker baseline in EXP-05 is
explained by the reference target itself. Uniform label-free enrollment naturally converges toward
the speaker's unlabeled marginal pitch median.

By K=10, the K-shot estimate is substantially closer to the marginal center than the neutral
reference in all three controlled datasets.

## Functional Result

The full marginal speaker reference is not merely a numerical alternative. Relative pitch defined
around the marginal center performs as well as, or better than, neutral-reference relative pitch.
On ESD and MEAD it is significantly better by approximately 2.0 and 1.4 macro-F1 points.

## Interpretation

Speaker-relative prosody does not require a privileged semantic neutral anchor. What matters may be
removing a stable speaker-specific location in acoustic space. This makes label-free enrollment a
natural reference-frame construction rather than a noisy approximation to a neutral-emotion oracle.

## Boundary

The full marginal oracle uses all eligible speaker utterances, including evaluation utterances, and
is therefore diagnostic only. Deployment claims must use finite K-shot enrollment as in EXP-05.

## Decision

mixed: the marginal-target mechanism is strongly supported, while the preregistered strict
functional-equivalence criterion fails because marginal references can outperform neutral references.
