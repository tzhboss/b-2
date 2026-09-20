# Experiment Report — EXP-20260920-04

## Registered Hypothesis
WavLM-large would show layer-dependent organization of absolute, speaker-relative, and speaker-baseline pitch information, and fixed middle layer 12 might recover a larger explicit Relative-minus-Absolute emotion effect than final layer 24.

## Main Results
All three pitch targets are strongly linearly decodable from WavLM hidden states, especially in early/middle layers.

Peak R2:
- ESD: absolute 0.958, relative 0.895, implied baseline 0.983.
- MEAD: absolute 0.987, relative 0.861, implied baseline 0.971.
- RAVDESS: absolute 0.939, relative 0.892, implied baseline 0.979.

Relative-pitch decodability drops substantially toward the final block:
- ESD: middle block 5-12 = 0.8825; final block 21-24 = 0.8114.
- MEAD: 0.8462 -> 0.7609.
- RAVDESS: 0.8766 -> 0.7754.

The preregistered layer-structure criterion is therefore supported.

However, explicit Relative-minus-Absolute emotion gains remain small even at layer 12:
- ESD: +0.130 percentage points.
- MEAD: +0.009 percentage points.
- RAVDESS: +0.159 percentage points.

These are not large enough to satisfy the preregistered middle-layer conditional-effect criterion.

## Supported Claim
WavLM retains highly decodable absolute pitch, speaker-relative pitch, and implied speaker-baseline information, but the accessibility of these quantities is layer-dependent. Speaker-relative pitch is most accessible in early/middle layers and degrades more strongly in later layers.

## Unsupported Stronger Claim
This experiment does not show that middle-layer WavLM restores the large Relative-versus-Absolute emotion advantage seen in prosody-only probes.

## Interpretation
The final-layer convergence observed in EXP-20260920-03 is not because WavLM lacks reference-frame information. Instead, the network contains strong absolute, relative, and baseline pitch information internally, while explicit pitch augmentation adds little to the final emotion classifier once the learned speech representation is available.

## Decision
valid / inconclusive for the primary middle-layer conditional-effect hypothesis; preregistered layer-dependent pitch structure is supported.

## Next Step
Move from oracle speaker reference frames to unseen-speaker reference estimation. Measure how many enrollment utterances are needed to recover useful relative pitch without target leakage.
