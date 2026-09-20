# Experiment Report — EXP-20260920-16

## Main Result

Continuous VAD regression produces a different reference-frame pattern from the controlled
categorical-emotion experiments.

On MSP-Podcast, Absolute prosody consistently outperforms speaker-relative prosody for Arousal and
Dominance. The difference is largest for all-attribute and loudness representations.

Valence is essentially not predictable from these three low-dimensional prosodic attribute sets:
CCC remains near zero under every representation.

## Interpretation

The strong Arousal/Dominance baseline restoration shows that speaker-stable prosodic information is
not merely nuisance for continuous affect ratings in MSP. Relative-only normalization removes a
large amount of useful information.

The result suggests that raw VAD ratings may contain both:
1. between-speaker affective priors or stable production differences; and
2. within-speaker utterance-level affective deviations.

The next experiment should decompose the VAD target itself into speaker mean plus within-speaker
residual and test which acoustic reference-frame component predicts each part.

## Decision

mixed.
