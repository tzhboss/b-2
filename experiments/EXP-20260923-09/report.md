# Experiment Report — EXP-20260923-09

## Result

The published PLDC calibration mechanism produces sharply different component profiles on the unchanged standard absolute Arousal/Dominance task.

For Arousal, k=10 mu-shift gives Overall +0.03155 and Between +0.04925, while Within is numerically zero. The corresponding sigma-shift gives Overall +0.01142, Between approximately zero, and Within +0.06678. For Dominance, k=10 mu-shift gives Overall +0.03104 / Between +0.05420 / Within approximately zero; sigma-shift gives Overall +0.02375 / Between approximately zero / Within +0.09009. Combined mu+sigma improves both components.

The pattern is stable for k=5,10,20,50. k=1 is an informative boundary condition: nearest-speaker mu calibration reduces Between by -0.06102 for Arousal and -0.06374 for Dominance, showing that adding speaker calibration can miscalibrate when the reference estimate is poor.

All paired confidence intervals use speaker-cluster bootstrap over 1,915 speakers and average the same resampled speakers across three modeling seeds. Retrieval audit contains 5,745 unique seed/fold/speaker rows with no duplicate keys.

## Interpretation

This is direct evidence that a single Overall CCC conflates mechanistically distinct improvements. PLDC mu calibration changes speaker-level calibration without changing within-speaker centered tracking; sigma calibration changes within-speaker variation without changing speaker means. Both can improve Overall, but they improve different capabilities.

This strengthens Route A beyond the synthetic Relative/Hybrid diagnostic: a published personalization mechanism itself contains separable Between and Within interventions whose Overall gains have different origins.

## Decision

pass; strong support for component attribution.
