# Experiment Report — EXP-20260920-04

## Main Result

WavLM-large retains highly decodable absolute pitch, speaker-relative pitch, and implied speaker pitch baseline across its hidden states, but the information profile changes with depth.

Relative-pitch R2 peaks in the early/middle network and declines toward the final block:
- ESD block means: 0-4=0.873, 5-12=0.882, 13-20=0.835, 21-24=0.811.
- MEAD: 0.827, 0.846, 0.788, 0.761.
- RAVDESS: 0.843, 0.877, 0.825, 0.775.

Absolute pitch remains even more linearly decodable, while implied speaker baseline is extremely decodable throughout the network, with peak R2 around 0.97-0.98.

## Interpretation

WavLM does not simply discard prosodic reference-frame information. Stable speaker baseline and utterance-relative pitch are both represented strongly, especially in early and middle layers. Later layers reorganize or compress relative-pitch information, but do not erase it.

However, explicitly appending Relative rather than Absolute pitch to WavLM layer 12 does not recover the large prosody-only emotion advantage. The observed layer-12 Relative-minus-Absolute deltas are only about 0.01-0.16 percentage points.

## Supported Claim

Prosodic reference-frame information has a clear layer-dependent representation profile inside WavLM-large.

## Unsupported Stronger Claim

This experiment does not support the claim that a middle WavLM layer restores the large explicit Relative-versus-Absolute downstream emotion advantage seen in prosody-only probes.

## Decision

mixed: representation-structure hypothesis supported; conditional-emotion hypothesis inconclusive.
