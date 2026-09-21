# Experiment Audit — EXP-20260921-10

## Runtime integrity

- Source/runtime script and config SHA256 values match.
- Immutable /tmp execution used.
- 10,039 IEMOCAP utterances, 10 speakers.
- 4,050 / 4,050 fold metric rows complete.
- 90 / 90 effect curves complete.
- 36 / 36 slope tests complete.
- No metric NaNs.
- relative_db and rms are excluded from all prediction feature sets.

## Pitch+Rate Relative-minus-Absolute slopes

Linear Ridge:
- Arousal: -0.02245, 95% CI [-0.03733, -0.00846].
- Dominance: -0.01089, [-0.01675, -0.00350].

Quadratic Ridge:
- Arousal: -0.02646, [-0.04277, -0.01133].
- Dominance: -0.01752, [-0.02431, -0.00992].

All four semantically safe external-replication slopes are significantly negative.

## Lambda=0 endpoints

Pitch+Rate Relative-minus-Absolute:
- Linear Arousal: +0.13633.
- Linear Dominance: +0.04577.
- Quadratic Arousal: +0.15514.
- Quadratic Dominance: +0.04862.

Within-speaker endpoint replication is fully supported.

## Attribute evidence

Linear Ridge:
- Pitch Arousal slope: -0.01900, significant.
- Pitch Dominance: unresolved.
- Rate Arousal: -0.00853, significant.
- Rate Dominance: -0.01228, significant.

The preregistered attribute criterion is supported.

## Consequence for IEMOCAP evidence

IEMOCAP remains a valid external replication of the target-reference mechanism using pitch and
speaking rate. It must not be used as main evidence for loudness because EXP-20260921-09 rejected
loudness-definition robustness.

## Verdict

- Validity: valid.
- Decision: pass.
