# Experiment Report — EXP-20260921-06

## Main Result

Speaker-center information helps raw VAD because it is aligned to the correct speaker, not merely
because the model receives additional center-valued dimensions.

On MSP, permuting center identity removes roughly 0.30 CCC of raw Arousal performance and 0.24 CCC
of raw Dominance performance relative to the correctly aligned Hybrid representation.

The same permutation has essentially no effect after VAD is centered within speaker.

## Interpretation

This directly supports the information-matching mechanism:
- raw targets containing speaker-level structure can use the corresponding speaker acoustic prior;
- within-speaker targets cannot use that prior once speaker-level target structure is removed.

## Decision

pass.
