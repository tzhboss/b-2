# Experiment Report — EXP-20260920-17

## Main Result

The apparent contradiction between raw MSP VAD and categorical controlled-corpus emotion is
explained by decomposing the target itself.

Raw Arousal and Dominance contain strong between-speaker components. Stable prosodic baselines
predict these speaker-level affect means well. Therefore Absolute prosody can outperform Relative
prosody on raw VAD because the target itself rewards between-speaker information.

After subtracting each speaker's VAD mean, the result reverses: Relative pitch/loudness/all-prosody
strongly outperform Absolute for within-speaker Arousal and Dominance residuals, while adding the
speaker baseline back has essentially zero effect.

## Interpretation

Reference-frame choice should match the target reference frame.

For raw population-level VAD:
- stable speaker baseline is task-relevant;
- Absolute/Hybrid can win.

For within-speaker affective deviation:
- stable speaker baseline becomes nuisance;
- Relative becomes the appropriate representation.

This directly supports the broader information-redistribution account.

## Decision

pass.
