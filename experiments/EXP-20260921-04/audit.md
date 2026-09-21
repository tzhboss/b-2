# Experiment Audit — EXP-20260921-04

## Runtime integrity

- Source/runtime script SHA256 match.
- Source/runtime config SHA256 match.
- Immutable /tmp execution used.
- 2,700 / 2,700 fold-level metric rows complete.
- 60 / 60 effect-curve rows complete.
- 24 / 24 slope rows complete.
- No metric NaNs.

## Relative-minus-Absolute slope versus lambda

Linear Ridge:
- MSP Arousal: -0.4714, 95% CI [-0.4816, -0.4597].
- MSP Dominance: -0.3756, [-0.3848, -0.3652].
- IEMOCAP Arousal: -0.0340, [-0.0491, -0.0200].
- IEMOCAP Dominance: -0.0234, [-0.0312, -0.0157].

Quadratic Ridge:
- MSP Arousal: -0.4702, 95% CI [-0.4806, -0.4584].
- MSP Dominance: -0.3730, [-0.3819, -0.3630].
- IEMOCAP Arousal: -0.0394, [-0.0573, -0.0227].
- IEMOCAP Dominance: -0.0312, [-0.0404, -0.0221].

All eight preregistered Arousal/Dominance corpus-by-model slopes are significantly negative.

## Lambda=0 endpoint

Relative-minus-Absolute is positive for Arousal and Dominance in both corpora under both model
families. The endpoint criterion is fully supported.

## Model-family consistency

All eight Arousal/Dominance slope signs agree with the target-reference direction.

The Hybrid-minus-Relative slope itself is not model-invariant in IEMOCAP: quadratic Ridge can
extract additional nonlinear information without explicit center exposure. This does not affect
the preregistered primary Relative-versus-Absolute mechanism test.

## Verdict

- Validity: valid.
- Decision: pass.
- Target-reference coupling is robust across the two fixed model families.
