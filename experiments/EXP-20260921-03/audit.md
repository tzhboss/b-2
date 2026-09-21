# Experiment Audit — EXP-20260921-03

## Runtime integrity

- Source/runtime script and config SHA256 values match.
- Execution used immutable /tmp copies.
- Output schema matches preregistration.

## Completeness

- MSP: 197,014 rows, 1,915 speakers.
- IEMOCAP: 10,039 rows, 10 speakers.
- Fold metrics: 1,800 / 1,800.
- Summary rows: 120 / 120.
- Effect-curve rows: 30 / 30.
- Slope rows: 12 / 12.
- Metric NaNs: zero.

## Relative-minus-Absolute slopes versus lambda

Arousal:
- MSP: -0.4714, 95% CI [-0.4816, -0.4597].
- IEMOCAP: -0.0340, [-0.0491, -0.0200].

Dominance:
- MSP: -0.3756, [-0.3848, -0.3652].
- IEMOCAP: -0.0234, [-0.0312, -0.0157].

The preregistered target-reference coupling criterion is fully supported.

## Hybrid-minus-Relative slopes versus lambda

Arousal:
- MSP: +0.3032, 95% CI [0.2929, 0.3119].
- IEMOCAP: +0.0108, [-0.0024, 0.0225].

Dominance:
- MSP: +0.2420, [0.2339, 0.2495].
- IEMOCAP: +0.0151, [0.0060, 0.0239].

The all-cells center-utility criterion is not fully supported because IEMOCAP Arousal crosses zero.

## Lambda=0 endpoints

Relative-minus-Absolute at pure within-speaker target:
- MSP Arousal: +0.1952.
- MSP Dominance: +0.1431.
- IEMOCAP Arousal: +0.1113.
- IEMOCAP Dominance: +0.0336.

The endpoint criterion is fully supported.

## Verdict

- Validity: valid.
- Decision: mixed.
- Controlled target-reference coupling: supported.
- Center-utility coupling: strongly supported except for IEMOCAP Arousal, which is directionally positive but unresolved.
