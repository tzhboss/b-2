# Experiment Audit — EXP-20260921-06

## Runtime integrity

- Source/runtime script and config SHA256 values match.
- Immutable /tmp runtime used.
- 2,160 metric rows complete.
- 180 fold-level permutation-effect rows complete.
- 12 summary cells complete.
- No metric NaNs.

## True center versus speaker-permuted center

MSP raw VAD:
- Arousal: +0.2955 CCC, 95% CI [0.2856, 0.3039].
- Dominance: +0.2363, [0.2285, 0.2436].
- Valence: +0.0052.

MSP residual VAD:
- Arousal: +0.00139.
- Dominance: +0.00100.
- Valence: +0.00006.

IEMOCAP raw VAD:
- Arousal: +0.0130, CI narrowly crosses zero.
- Dominance: +0.0137, CI above zero.
- Valence: approximately zero.

IEMOCAP residual VAD:
- Arousal: +0.00078.
- Dominance: +0.00038.
- Valence: approximately zero.

## Registered Criteria

- MSP raw center-identity utility: supported.
- Residual center irrelevance: supported in both corpora.
- MSP raw-versus-residual contrast: supported.
- Rejection criterion: not met.

## Interpretation

The Hybrid benefit on raw MSP Arousal/Dominance requires the correct speaker-center identity.
Preserving the center marginal distribution but assigning centers to the wrong speakers destroys
most of the benefit.

Once the target is within-speaker centered, correct center identity becomes essentially irrelevant.

## Verdict

- Validity: valid.
- Decision: pass.
