# Experiment Report — EXP-20260920-05

## Main Result

Label-free K-shot speaker references are sufficient to recover most of the emotion utility of
oracle speaker-relative pitch on completely unseen speakers.

At K=10, K-shot Relative minus matched Absolute macro-F1 is:
- ESD: +0.1065.
- MEAD: +0.0980.
- RAVDESS: +0.0744.

All three paired-bootstrap confidence intervals are strictly above zero.

K=10 K-shot Relative is statistically indistinguishable from Oracle-relative in all three
datasets under the registered paired comparison.

## Unexpected Result

The estimated K-shot speaker center does not converge monotonically to the existing neutral-derived
oracle baseline. ESD is the clearest counterexample: baseline MAE relative to the neutral oracle is
lower at K=1 than at K=10, even though downstream K=10 emotion performance nearly matches the oracle.

## Interpretation

The neutral-derived reference is therefore not necessarily the unique or even the natural target
for label-free reference estimation. A random label-free enrollment median may be estimating a
different speaker-specific center that is still highly useful for emotion classification.

This motivates a direct follow-up comparing the neutral reference with the speaker's unlabeled
marginal pitch center.

## Decision

mixed: the preregistered neutral-baseline convergence criterion is not supported, but the two
downstream utility/recovery criteria are strongly supported.
