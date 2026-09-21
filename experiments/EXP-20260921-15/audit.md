# Experiment Audit — EXP-20260921-15

## Purpose

Replace fold/seed-level uncertainty with speaker-cluster inference on complete out-of-fold
predictions.

## Runtime integrity

- One fixed speaker-disjoint split seed.
- One OOF prediction per utterance for every representation/target/lambda/model condition.
- Bootstrap unit: complete speaker.
- 5,000 speaker-cluster bootstrap replicates.
- Optimized sufficient-statistics implementation is mathematically equivalent to resampling full
  speaker utterance blocks and recomputing weighted CCC.

## MSP results

Linear Ridge:
- Arousal slope: -0.3188, 95% CI [-0.3384, -0.2998].
- Dominance slope: -0.1825, [-0.1969, -0.1683].
- lambda=0 Arousal Relative-Absolute: +0.1390, [0.1334, 0.1444].
- lambda=0 Dominance: +0.0727, [0.0685, 0.0769].

Quadratic Ridge:
- Arousal slope: -0.3214, 95% CI [-0.3411, -0.3022].
- Dominance slope: -0.1851, [-0.2000, -0.1705].
- lambda=0 Arousal: +0.1390, [0.1331, 0.1446].
- lambda=0 Dominance: +0.0746, [0.0702, 0.0792].

All four MSP slopes remain strongly below zero under speaker-cluster inference.

## IEMOCAP results

Linear Ridge:
- Arousal slope: -0.0192, 95% CI [-0.0662, +0.0326].
- Dominance slope: -0.0162, [-0.0484, +0.0123].

Quadratic Ridge:
- Arousal slope: -0.0281, 95% CI [-0.0758, +0.0256].
- Dominance slope: -0.0271, [-0.0617, +0.0058].

All four point estimates retain the negative direction, but none of the speaker-cluster slope CIs
exclude zero with only 10 speakers.

The pure within-speaker lambda=0 Relative advantage remains positive with speaker-cluster CIs
strictly above zero for all four IEMOCAP model-target cells.

## Registered criteria

- MSP speaker-cluster replication: supported.
- IEMOCAP direction replication: supported.
- Pure within-speaker endpoint replication: supported.
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: pass, with an important statistical boundary.

The strongest inferential evidence is MSP, where 1,915 speakers support narrow speaker-cluster
intervals. IEMOCAP independently preserves the effect direction and within-speaker endpoint but is
underpowered for slope significance at only 10 speakers.
