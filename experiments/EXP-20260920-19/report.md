# Experiment Report — EXP-20260920-19

## Main Result

Ten unlabeled enrollment utterances are enough to estimate a useful three-dimensional speaker
acoustic center for raw MSP VAD prediction.

K=10 reduces center-estimation error by roughly 68-71% relative to K=1 and yields Hybrid
Arousal/Dominance performance very close to a diagnostic full-speaker marginal-center oracle.

## Deployment Result

Compared with Absolute prosody:
- Arousal improves by +0.0194 CCC.
- Dominance improves by +0.0092 CCC.
- Valence improves by +0.0138 CCC.

Dominance narrowly misses the preregistered +0.010 deployment threshold, so the experiment is
formally mixed rather than pass.

## Interpretation

The stable speaker component needed for raw VAD does not require emotion or VAD labels at
deployment time. A small unlabeled enrollment set can estimate an acoustic reference that recovers
most of the full marginal-speaker Hybrid advantage.

## Decision

mixed.
