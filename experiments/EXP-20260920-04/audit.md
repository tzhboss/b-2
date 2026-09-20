# Experiment Audit — EXP-20260920-04

## Identity

- Experiment ID: EXP-20260920-04
- Runtime preregistration commit: ebaea6b
- Protocol: configs/protocols/wavlm_layerwise_pitch_v1.yaml
- Experiment config: configs/experiments/EXP-20260920-04.yaml

## Embedding Audit

- 50,669 unique samples.
- 25 hidden states per sample, each 1024 dimensions.
- Seven extraction shards, all exit code 0.
- All values finite.
- All shard checkpoint hashes match WavLM-large used in EXP-03.
- All shards ran in eval mode with gradients disabled.

## Evaluation Audit

- Pitch decodability fold rows: 2,700 / 2,700 expected.
- Pitch summary rows: 225 / 225 expected.
- Emotion fold rows: 288 / 288 expected.
- Emotion summary rows: 24 / 24 expected.
- Emotion paired-delta rows: 24 / 24 expected.
- Metric NaNs: zero.
- Same frozen layers 12 and 24 were used as preregistered.

## Registered Criterion Evaluation

Layer-dependent pitch structure is supported.

For relative-pitch decodability, the maximum preregistered block-level R2 difference is:
- ESD-English: 0.0711.
- MEAD-part0: 0.0853.
- RAVDESS-speech: 0.1012.

Thus at least one pitch target changes by more than 0.05 across preregistered layer blocks in all three datasets.

The middle-layer conditional emotion hypothesis is not supported:
- ESD Relative-minus-Absolute: layer 12 +0.00130 vs layer 24 +0.00038.
- MEAD: +0.00009 vs -0.00012.
- RAVDESS: +0.00159 vs +0.00056.

No two datasets meet the registered >=0.0025 layer-12-over-layer-24 difference with layer-12 CI above zero. The strict rejection criterion is also not met because RAVDESS differs by slightly more than 0.001.

## Audit Verdict

- Validity: valid.
- Decision: mixed.
- Layer-dependent pitch structure: supported.
- Middle-layer conditional reference-frame effect: inconclusive.
