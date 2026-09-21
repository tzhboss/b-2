# Experiment Audit — EXP-20260921-15

## Integrity

- One fixed speaker-disjoint OOF split.
- Speaker is the resampling unit.
- 5,000 speaker-cluster bootstrap replicates.
- MSP: 197,020 utterances, 1,915 speakers.
- IEMOCAP: 10,039 utterances, 10 speakers.
- Runtime script/config hashes match.
- Initial slow implementation was interrupted before producing scientific outputs; protocol and
  inferential definition were unchanged. The completed implementation uses exact speaker-level
  sufficient statistics to vectorize the same cluster bootstrap.

## MSP speaker-cluster results

Relative-minus-Absolute slope versus lambda:

- Linear Arousal: -0.3188, 95% CI [-0.3384, -0.2998].
- Linear Dominance: -0.1825, [-0.1969, -0.1683].
- Quadratic Arousal: -0.3214, [-0.3411, -0.3022].
- Quadratic Dominance: -0.1851, [-0.2000, -0.1705].

All four speaker-cluster CIs are strictly below zero.

At lambda=0:
- Linear Arousal: +0.1390, 95% CI [0.1334, 0.1444].
- Linear Dominance: +0.0727, [0.0685, 0.0769].
- Quadratic Arousal: +0.1390, [0.1331, 0.1446].
- Quadratic Dominance: +0.0746, [0.0702, 0.0792].

## IEMOCAP speaker-cluster results

Mean slopes remain negative:

- Linear Arousal: -0.0192, CI [-0.0662, +0.0326].
- Linear Dominance: -0.0162, [-0.0484, +0.0123].
- Quadratic Arousal: -0.0281, [-0.0758, +0.0256].
- Quadratic Dominance: -0.0271, [-0.0617, +0.0058].

With only ten speakers, all slope CIs cross zero. This corrects the earlier fold-level inference:
IEMOCAP supports the same direction but not a statistically precise slope estimate.

At lambda=0, however, Relative remains significantly better than Absolute for Arousal and
Dominance under both models.

## Registered criteria

- MSP speaker-cluster replication: supported.
- IEMOCAP direction replication: supported.
- Pure within-speaker endpoint replication: supported.
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: pass.

Primary paper inference should use these speaker-cluster intervals instead of fold/seed bootstrap
intervals.
