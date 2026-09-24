# EXP-20260924-14 report

## Decision
valid / pass

## Main result
On the identical nested 5-session-LOSO absolute SER OOF set, component-balanced maximin retains positive Overall, Between, and Within deltas versus nested best-single for both Arousal and Dominance.

Against the published PLDC mu+sigma baseline:
- Arousal Overall: +0.040261 CCC, 95% CI [0.016472, 0.067715]
- Arousal Within: +0.024427, 95% CI [0.003863, 0.045868]
- Dominance Overall: +0.094498, 95% CI [0.005147, 0.179274]

Against the strong inner-CV convex SSL ensemble, maximin is statistically indistinguishable on all three components for both targets; point Overall differences are -0.000402 (Arousal) and -0.002400 (Dominance).

## Interpretation
The decomposition-guided method does not establish superiority over a strong generic nested ensemble, but it substantially outperforms the published personalization/calibration baseline while preserving the intended joint component gains. This supports the claim that Overall/Between/Within decomposition can guide a competitive method design, not only post-hoc explanation.

Absolute SER remains the task; Between/Within remain diagnostic components.
