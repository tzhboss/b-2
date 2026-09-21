# Experiment Report — EXP-20260920-19

## Main Result

A small label-free enrollment set can recover most of the stable speaker information needed for
raw MSP Arousal/Dominance prediction.

By K=10, acoustic-center estimation error falls by roughly 68-71% across pitch, loudness, and
speaking rate. K-shot Hybrid improves over Absolute and approaches the full marginal-speaker oracle.

## Arousal

K=10 Hybrid improves over Absolute by +0.0194 CCC and is only 0.0053 below the marginal-oracle
Hybrid.

## Dominance

K=10 Hybrid improves over Absolute by +0.0092 CCC and is effectively at the marginal-oracle level
(-0.0005 CCC). The result is statistically positive but narrowly misses the preregistered +0.010
effect-size threshold.

## Interpretation

Deployment does not require emotion labels, neutral speech labels, or a full speaker history.
A small unlabeled reference set is sufficient to estimate a useful speaker acoustic center.
However, normalization alone is inappropriate for raw VAD: the estimated center must remain
available to the downstream model, as in the Hybrid representation.

## Decision

mixed.
