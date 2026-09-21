# Experiment Report — EXP-20260921-01

## Main Result

Unlabeled acoustic-center estimation keeps improving substantially after K=10, but downstream raw
VAD performance shows much stronger diminishing returns.

K=10 is extremely close to the preregistered practical plateau, but Arousal improves by 0.01028
CCC from K=10 to K=50, narrowly exceeding the <0.01 threshold. Dominance improves by 0.00946.

K=20 is descriptively closer to the practical plateau: K20-to-K50 gains are only 0.00564 for
Arousal and 0.00761 for Dominance.

## Interpretation

Accurate recovery of the physical acoustic center is not the same as downstream utility. A large
fraction of center-estimation error can remain without materially changing VAD prediction.

## Decision

mixed.
