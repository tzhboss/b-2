# Experiment Report — EXP-20260920-14

## Main Result

The Absolute-versus-Relative effect is only moderately stable across LogisticRegression and a
fixed HistGradientBoosting classifier.

Relative-minus-Absolute effect correlation is rho=0.564 and sign agreement is 70% among cells
with at least 0.01 Logistic effect magnitude. Baseline-addition effects are even less stable.

## Interpretation

Reference-frame preference is not purely a property of the acoustic variable and corpus.
Model family changes how much of the same one- or two-dimensional representation can be exploited.
The strongest controlled-corpus pitch effects remain directionally stable, but several natural-corpus
effects, especially MSP pitch/rate, attenuate or reverse under HGB.

## Boundary

This experiment uses one fixed nonlinear tree model and does not establish a universal
classifier-capacity law.

## Decision

inconclusive under preregistered robustness criteria.
