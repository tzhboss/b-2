# CURRENT_STATE.md

## Phase

Extended attributes, corrected speaker-cluster inference, and SSL-level reference-frame
intervention are complete.

## New strongest result

The target-reference mechanism replicates directly in frozen WavLM embedding space on IEMOCAP.

Layer 12:
- Arousal slope -0.0829, 95% CI [-0.1309, -0.0359].
- Dominance -0.0341, [-0.0583, -0.0118].

Layer 24:
- Arousal -0.0958, [-0.1429, -0.0482].
- Dominance -0.0455, [-0.0707, -0.0181].

At lambda=0 Relative embeddings are significantly better in all four cells; at lambda=1 the
difference reverses and Absolute becomes better.

## Remaining major limitation

A third genuinely independent continuous-affect corpus is still unavailable locally. External
breadth remains the main unresolved limitation, rather than attribute or model-family breadth.

## Next legal step

Update paper assets and main manuscript to promote EXP-20260921-16 and EXP-20260921-15 into the
central evidence stack; then reassess whether additional voice-quality extraction would materially
change the paper.
