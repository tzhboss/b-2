# Experiment Audit — EXP-20260920-19

## Completeness

- Eligible rows: 195,779.
- Speakers: 1,911.
- Fold-level metric rows: 1,260 / 1,260 expected.
- Paired-delta rows: 72 / 72 expected.
- Audit rows: 60 / 60 expected.
- Center-error summary rows: 12 / 12 expected.
- Metric NaNs: zero.
- Train/test speaker overlap: zero.
- Enrollment/evaluation row overlap: zero.

## K-shot center convergence

Mean absolute error to the full marginal speaker acoustic center decreases from K=1 to K=10 by:
- Pitch: 67.8%.
- Loudness: 68.7%.
- Log-rate: 70.6%.

The preregistered >=25% convergence criterion is satisfied for all three attributes.

## K=10 downstream utility

K-shot Hybrid minus Absolute:
- Arousal: +0.01944 CCC, 95% CI [0.01651, 0.02221].
- Dominance: +0.00923 [0.00739, 0.01091].
- Valence: +0.01377 [0.01290, 0.01476].

Arousal satisfies the preregistered +0.015 criterion.
Dominance is significantly positive but narrowly misses the preregistered +0.010 threshold by 0.00077.

## Oracle recovery

K=10 Hybrid minus marginal-oracle Hybrid:
- Arousal: -0.00529 CCC.
- Dominance: -0.00050 CCC.
- Valence: -0.00280 CCC.

Thus K=10 is within the preregistered 0.02 oracle gap for both Arousal and Dominance.

## Important boundary

K-shot Relative without baseline is strongly worse than Absolute for raw Arousal/Dominance:
- Arousal: -0.2734 CCC.
- Dominance: -0.2257 CCC.

The benefit comes from estimating and exposing the stable acoustic center, not from normalization alone.

## Verdict

- Validity: valid.
- Decision: mixed.
- Center convergence: supported.
- Arousal deployment recovery: supported.
- Dominance deployment recovery: strong positive evidence but misses the preregistered effect-size threshold narrowly.
- Marginal-oracle recovery: supported.
