# CURRENT_STATE.md

## Phase
Layer-wise WavLM experiment completed and audited; unseen-speaker reference-estimation experiment is next.

## Valid evidence
- EXP-20260920-02: valid/pass. Prosody-only pitch has a strong task-dependent Absolute/Relative reversal.
- EXP-20260920-03: valid/inconclusive. Final-layer WavLM largely removes incremental Absolute-vs-Relative pitch differences.
- EXP-20260920-04: valid/inconclusive for the primary conditional-effect hypothesis; preregistered layer-dependent pitch structure is supported.

## Main current interpretation
WavLM strongly contains absolute pitch, speaker-relative pitch, and implied speaker-baseline information.
Relative-pitch linear decodability is strongest in early/middle layers and decreases substantially in later layers,
but explicit Relative-vs-Absolute pitch augmentation remains small for emotion once WavLM features are present.

## Next legal step
Register and execute an unseen-speaker K-shot reference-estimation experiment with strict enrollment/target separation.
