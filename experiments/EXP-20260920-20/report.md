# Experiment Report — EXP-20260920-20

## Main Result

Label-free speaker acoustic-center estimation improves monotonically and substantially with more
enrollment audio. By K=20, raw Arousal and Dominance Hybrid performance is already almost identical
to the marginal-speaker oracle.

However, Arousal performance decreases from K=20 to K=50 even though the center estimate becomes
much more accurate. This cannot yet be interpreted as over-enrollment or true saturation because
larger K also removes more utterances from downstream training/evaluation.

## Next Diagnostic

Use a fixed 50-utterance reserved enrollment pool for every K. Keep the downstream train/test rows
identical and use only the first K nested enrollment utterances to estimate the speaker center.

## Decision

mixed.
