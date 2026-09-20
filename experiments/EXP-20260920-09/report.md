# Experiment Report — EXP-20260920-09

## Main Result

Reference-frame preference is strongly emotion-specific and corpus-dependent.

The clearest result is MSP. Relative pitch is better than Absolute for angry, happy, sad, and
surprise, but dramatically worse for neutral. The single neutral class is large enough to reverse
the corpus-level aggregate result.

MSP Relative-minus-Absolute speaker-balanced F1:
- angry: +0.0177.
- happy: +0.0539.
- sad: +0.1273.
- surprise: +0.0060.
- neutral: -0.3229.

Adding the stable speaker baseline back to Relative almost restores neutral performance:
+0.2685 F1 on neutral. It also helps surprise, but hurts angry, happy, and sad.

## Conflict Set

A robust sign-conflict set was defined without labels: an utterance is a conflict when its
absolute pitch direction relative to the train-speaker corpus reference is opposite to its
within-speaker relative-pitch direction, with both magnitudes at least 0.5 semitone.

These conflicts are common: approximately 19% to 32% of test utterances.

On conflict utterances:
- ESD, MEAD, and RAVDESS strongly favor Relative.
- MSP and MELD favor Absolute.

This shows that the corpus-level difference is not just a small average effect. When the two
reference frames explicitly disagree, the direction of useful information changes with domain.

## Interpretation

The MSP reversal is not evidence that within-speaker deviation is useless. For most MSP emotion
classes Relative is better. The reversal is dominated by neutral, whose recognition strongly
benefits from the stable speaker baseline removed by normalization.

This supports a stronger paper claim:
prosodic normalization redistributes task-relevant information, and the useful reference frame
depends jointly on corpus/domain and emotion class.

## Decision

pass.
