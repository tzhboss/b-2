# Experiment Audit — EXP-20260920-18

## Completeness

- Eligible all-attribute rows: 195,779.
- Speakers: 1,911.
- Minimum speaker support: 11.
- Train-only disagreement thresholds: 45 / 45 target × seed × fold rows.
- Fold/stratum metric rows: 1,080.
- Summary rows: 72.
- Delta rows: 48.
- Metric NaNs: zero.

## Low-Disagreement Robustness

Raw VAD, Relative+Baseline minus Relative:
- Arousal: +0.1581 CCC, 95% CI [0.1496, 0.1682].
- Dominance: +0.1298 [0.1229, 0.1367].
- Valence: +0.0047.

Within-speaker residual VAD, Relative minus Absolute:
- Arousal: +0.1148 CCC, 95% CI [0.1109, 0.1182].
- Dominance: +0.0924 [0.0879, 0.0973].
- Valence: +0.0054.

Thus both preregistered Arousal/Dominance mechanism-robustness criteria are supported.

## Label-Reliability Gradient

Best low-disagreement versus high-disagreement CCC:
- Arousal raw: +0.0258.
- Arousal residual: +0.0713.
- Dominance raw: -0.0288.
- Dominance residual: -0.0059.
- Valence raw: -0.0015.
- Valence residual: -0.0041.

Only one of six cells exceeds the preregistered +0.05 low-over-high threshold; the broad
label-reliability gradient criterion is not supported.

## Verdict

- Validity: valid.
- Decision: mixed.
- Raw baseline and within-speaker Relative mechanisms survive low human-annotation disagreement.
- A universal monotonic label-disagreement performance gradient is not supported.
