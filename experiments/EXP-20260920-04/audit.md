# Experiment Audit — EXP-20260920-04

## Identity
- Experiment ID: EXP-20260920-04
- Runtime commit: fe8ad19a082e6ebb995f668989272c7d8fccf518
- Protocol: configs/protocols/wavlm_layerwise_pitch_v1.yaml
- Config: configs/experiments/EXP-20260920-04.yaml

## Embedding Audit
- 50,669 unique sample IDs.
- 25 pooled hidden states per sample, each 1024-D.
- All values finite.
- Seven extraction shards completed successfully.
- Identical WavLM checkpoint SHA256 across shards.
- WavLM eval mode enabled and all parameters gradient-disabled.

## Evaluation Completeness
- Pitch decodability fold rows: 2,700 / 2,700 expected.
- Emotion fold rows: 288 / 288 expected.
- Emotion paired-delta rows: 24 / 24 expected.
- Dataset inventories: 3 / 3 expected.
- NaN metrics: zero.
- Seeds: 20260920, 20260921, 20260922.
- Folds: four per seed.
- Conditional layers were fixed at 12 and 24 before execution.

## Registered Conditional-Effect Criterion
Layer-12 Relative-minus-Absolute emotion macro-F1:
- ESD: +0.001303, CI [+0.000679,+0.001899].
- MEAD: +0.000085, CI [-0.000574,+0.000714].
- RAVDESS: +0.001592, CI [-0.001040,+0.004920].

Layer-24 Relative-minus-Absolute:
- ESD: +0.000380.
- MEAD: -0.000122.
- RAVDESS: +0.000556.

Layer-12 minus layer-24 effect differences are +0.000924 / +0.000207 / +0.001037 for ESD / MEAD / RAVDESS. None reach the preregistered +0.0025 threshold in two datasets. The strict rejection rule is also not met because RAVDESS exceeds the 0.001 equality threshold. Therefore the conditional-effect hypothesis is inconclusive.

## Registered Layer-Structure Criterion
Mean blockwise cross-validated R2 for relative pitch changes by:
- ESD: max-min = 0.0711.
- MEAD: max-min = 0.0853.
- RAVDESS: max-min = 0.1012.

This exceeds the preregistered 0.05 threshold in all three datasets. RAVDESS absolute-pitch block span is also 0.0562. Thus layer-dependent pitch structure is supported.

## Verdict
- Validity: valid.
- Overall decision field: inconclusive, because the primary middle-vs-final conditional effect is not established.
- Secondary preregistered finding: layer-dependent pitch structure supported.
