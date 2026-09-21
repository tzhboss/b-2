# Experiment Report — EXP-20260921-03

## Main Result

Reference-frame utility changes systematically when the target reference frame is manipulated
while acoustic inputs, speakers, folds, and model remain fixed.

For both MSP and IEMOCAP Arousal/Dominance, increasing the between-speaker component of the VAD
target makes Relative less favorable versus Absolute. At the pure within-speaker endpoint,
Relative is better in every Arousal/Dominance corpus-target cell.

MSP exhibits a full sign crossover because its speaker-level target structure is strong.
IEMOCAP shows a smaller but still significant decline because its raw VAD contains little
between-speaker variance.

## Interpretation

These results directly support a target-reference matching principle:

- within-speaker targets favor speaker-relative representations;
- adding between-speaker target structure increases the usefulness of stable speaker acoustic
  information;
- whether a raw target prefers Absolute, Relative, or Hybrid depends on how much between-speaker
  information the target itself contains.

## Decision

mixed.
