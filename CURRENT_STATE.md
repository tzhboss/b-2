# CURRENT_STATE.md

## Phase

Speaker-cluster inference completed and validated. WavLM embedding-level intervention is next.

## Strongest statistical result

MSP target-reference slopes remain strongly negative after speaker-cluster bootstrap:
- Linear Arousal -0.3188, Dominance -0.1825.
- Quadratic Arousal -0.3214, Dominance -0.1851.
All 95% CIs are below zero.

IEMOCAP retains negative point estimates under both models but slope CIs cross zero with only
10 speakers. Its lambda=0 Relative advantage remains significantly positive.

## Extended attributes

F0 variability, pause ratio, and voiced ratio also show significant target-reference coupling on MSP.

## Next legal step

Repair EXP-20260921-16 extraction using a safe Torch/model-loading environment and run the
embedding-level target-reference intervention.
