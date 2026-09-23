# Experiment Report — EXP-20260923-08

## Result

EXP-20260923-06 was rerun with Ridge solver=lsqr and tol=1e-10 on exactly the same frozen WavLM
embeddings and five leave-one-session-out folds.

Mean Relative-minus-Absolute lambda slopes remain negative in all four cells, with 5/5 negative
held-out sessions in every cell:

- Layer 12 Arousal: -0.0783263.
- Layer 12 Dominance: -0.0297663.
- Layer 24 Arousal: -0.0895352.
- Layer 24 Dominance: -0.0358523.

Absolute changes in the mean slope relative to the default Ridge solver are tiny:
- L12 A: 0.00000050
- L12 D: 0.00000090
- L24 A: 0.00000291
- L24 D: 0.00000137

All are far below the preregistered 0.005 tolerance.

## Interpretation

The session-disjoint WavLM reference-frame result is numerically solver-stable. The
ill-conditioned-matrix warnings observed in the default solver do not explain the directional
effect.

## Decision

pass.
