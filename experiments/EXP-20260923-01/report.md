# Experiment Report — EXP-20260923-01

## Question

For the ordinary absolute MSP Arousal/Dominance task, do conventional Overall CCC scores hide
different Between-speaker calibration and Within-speaker affect-tracking behavior?

## Result

The preregistered strong-support pattern was **not** observed. In particular, Relative did not
improve the Within diagnostic over Absolute. It was slightly worse in every model/target cell.

However, the diagnostic decomposition reveals a different and potentially important pattern:
Hybrid's Overall improvement over Absolute is driven almost entirely by Between-speaker
calibration, while Within-speaker tracking changes little.

Linear Ridge:
- Arousal: Hybrid-Absolute Overall +0.0539; Between +0.0902; Within +0.0001.
- Dominance: Hybrid-Absolute Overall +0.0378; Between +0.0725; Within -0.0020.

Quadratic Ridge:
- Arousal: Hybrid-Absolute Overall +0.0570; Between +0.0881; Within +0.0012.
- Dominance: Hybrid-Absolute Overall +0.0433; Between +0.0767; Within -0.0049.

Relative strongly removes Between calibration:
- Linear Arousal Relative-Absolute Between -0.7070.
- Linear Dominance -0.6704.
- Quadratic Arousal -0.7217.
- Quadratic Dominance -0.6830.

But Relative also reduces Within CCC modestly (roughly -0.019 to -0.037), so this experiment does
not support a generic claim that speaker normalization improves within-person affect tracking.

For Dominance, the Overall winner is Hybrid while the Within winner is Absolute under both Linear
and Quadratic Ridge, giving an explicit ranking change across evaluation views. For Arousal,
Hybrid is the point-estimate winner for both Overall and Within, but its Within advantage over
Absolute is negligible and the paired CI includes zero.

## Interpretation

This supports Route A only in a revised form: Overall improvements can be dominated by
Between-speaker calibration gains without improving Within-speaker tracking. The result does not
support the stronger preregistered hypothesis that Relative representations generally trade
Between performance for improved Within performance.

The next critical test is cross-model: determine whether different speech representations/backbones
with similar Overall performance have different Between/Within profiles and whether model rankings
change under the Within diagnostic.

## Decision

informative_mixed.
