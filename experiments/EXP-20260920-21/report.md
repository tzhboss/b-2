# Experiment Report — EXP-20260920-21

## Main Result

With downstream rows fixed across all K, label-free acoustic-reference estimation continues to
improve through K=50, while raw MSP VAD Hybrid performance is already practically saturated by
K=20.

K=20 is within 0.0043 CCC of the marginal oracle for Arousal and essentially identical for
Dominance. Moving from K=20 to K=50 yields only +0.0043 Arousal CCC and +0.0001 Dominance CCC.

## Interpretation

The apparent K20-to-K50 Arousal drop in EXP-20 was caused by the changing downstream sample pool,
not by over-enrollment. Once that confound is removed, performance increases modestly with K.

For practical deployment, approximately 20 unlabeled utterances provide most of the usable
speaker-reference benefit on this MSP cohort.

## Decision

mixed: practical saturation supported; strict diminishing-return curve not supported.
