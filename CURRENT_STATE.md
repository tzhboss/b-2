# CURRENT_STATE.md

## Phase

Layer-wise frozen WavLM pitch-structure experiment registered; execution pending.

## Active research question

Where across WavLM-large layers are absolute pitch, speaker-relative pitch, and speaker pitch
baseline linearly decodable, and is the Relative-versus-Absolute emotion effect larger at a
fixed middle layer than at the final layer?

## Valid evidence

- EXP-20260920-02: valid, pass; strong prosody-only reference-frame interaction.
- EXP-20260920-03: valid, inconclusive; final-layer WavLM largely removes the incremental
  Absolute-versus-Relative pitch difference, while explicit Hybrid/all-prosody can still help emotion.

## Recent key experiments

- EXP-20260920-01 — completed, invalid, inconclusive.
- EXP-20260920-02 — completed, valid, pass.
- EXP-20260920-03 — completed, valid, inconclusive.
- EXP-20260920-04 — planned layer-wise pitch decodability and middle-vs-final emotion probe.

## Next legal step

Extract all 25 masked-mean pooled hidden states from the fixed WavLM-large checkpoint using the
audited EXP-03 audio manifest, then execute the registered layer-wise probes and audit them.
