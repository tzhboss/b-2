# Experiment Audit — EXP-20260920-16

## Data and provenance

- VAD targets are MSP human SAM 1-7 ratings from labels_detailed.csv, aggregated by utterance mean.
- They are not model-generated labels.
- All retained rows exclude speaker_id Unknown.
- Only speaker_neutral or speaker_neutral_shrunk reference scopes are used.
- Minimum retained speaker support is 11 rows.
- Mean annotator count is approximately 5.49 per utterance.

## Completeness

- Fold-level metric rows: 720 / 720 expected.
- Summary rows: 48 / 48 expected.
- Paired-delta rows: 48 / 48 expected.
- Inventory rows: 4 / 4 expected.
- Metric NaNs: zero.
- Duplicate evaluation keys: zero.

## Main Reference-Frame Effects

Relative-minus-Absolute speaker-balanced CCC:

Arousal:
- Pitch: -0.0383.
- Loudness: -0.0968.
- Rate: -0.0214.
- All: -0.1301.

Dominance:
- Pitch: -0.0157.
- Loudness: -0.0993.
- Rate: -0.0112.
- All: -0.1146.

Valence:
- Effects are very small positive values, from +0.0007 to +0.0046, while absolute CCC itself is approximately zero.

## Baseline Restoration

Relative+Baseline minus Relative CCC:

Arousal:
- Pitch: +0.0694.
- Loudness: +0.1027.
- Rate: +0.0546.
- All: +0.1629.

Dominance:
- Pitch: +0.0390.
- Loudness: +0.1044.
- Rate: +0.0233.
- All: +0.1310.

Valence:
- Small gains only, all below +0.005.

## Registered Criteria

- Arousal Relative-reference utility: not supported; all four effects are negative.
- VAD target-specificity: supported in all four attribute sets.
- Baseline redistribution: strongly supported.
- Global no-effect rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: mixed.
- The preregistered Arousal-relative hypothesis is contradicted.
- Strong target-specificity and baseline-redistribution effects are supported.
