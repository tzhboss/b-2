# Experiment Report — EXP-20260921-07

## Main Result

The target-reference mechanism survives deployment-style K=20 unlabeled reference estimation
strongly on MSP and directionally on IEMOCAP.

MSP K-shot slopes closely track the full-speaker-oracle slopes. IEMOCAP retains the same negative
direction but uncertainty grows because only ten speakers are available and K-shot center
estimation adds reference noise.

## Interpretation

The mechanism itself is not restricted to oracle speaker centers. On a large-speaker corpus,
twenty unlabeled utterances are sufficient to recover the target-reference coupling.

The next diagnostic should vary K on a fixed IEMOCAP downstream pool to distinguish insufficient
reference quality from genuinely absent deployment coupling.

## Decision

mixed.
