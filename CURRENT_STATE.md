# CURRENT_STATE.md

## Phase

Independent IEMOCAP VAD validation completed and audited.

## Main cross-corpus finding

MSP and IEMOCAP differ sharply in between-speaker target structure:
- MSP raw Arousal/Dominance have large between-speaker components and benefit from speaker baseline.
- IEMOCAP raw VAD has little between-speaker structure and already favors Relative for Arousal.
- Within-speaker residual Arousal/Dominance favor Relative in both corpora, with negligible
  baseline utility after target centering.

## Next legal step

Quantify target-structure coupling across MSP and IEMOCAP: test whether VAD target ICC predicts
raw Relative-minus-Absolute and baseline-addition utility across corpus-target cells.
