# Experiment Report — EXP-20260921-08

## Main Result

Increasing K substantially improves physical speaker-center estimation on IEMOCAP, but the
target-reference coupling statistic does not improve monotonically.

K=50 gives the closest deployment slopes to the full-speaker oracle for Arousal and Dominance,
while K=100 has lower center MAE but a larger slope gap.

## Interpretation

Acoustic-center estimation error is not the only source of uncertainty. With only ten speakers,
which enrollment utterances are selected can change the downstream center-target relationship even
when the estimated physical center becomes more accurate.

Therefore no claim of monotonic mechanism recovery with K is justified.

## Decision

mixed.
