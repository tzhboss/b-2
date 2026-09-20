# CURRENT_STATE.md

## Phase

MSP VAD target decomposition completed and audited.

## Main valid finding

Raw MSP Arousal/Dominance mix large between-speaker affective priors with within-speaker state.
Stable prosodic baselines predict the speaker-level component, explaining why Absolute/Hybrid
outperform Relative on raw VAD. Once the speaker VAD mean is removed from the target, Relative
pitch/loudness/all-prosody strongly outperform Absolute and baseline restoration becomes negligible.

## Next legal step

Merge preserved MSP per-annotator VAD disagreement statistics and test whether these reference-frame
effects survive low- versus high-disagreement human labels.
