# Experiment Audit — EXP-20260920-09

## Completeness

- Per-emotion fold rows: 1,500 / 1,500 expected.
- Per-emotion summary rows: 100 / 100 expected.
- Per-emotion paired-delta rows: 75 / 75 expected.
- Conflict fold rows: 75 / 75 expected.
- Conflict summary rows: 5 / 5 expected.
- Inventory rows: 5 / 5 expected.
- Metric NaNs: zero.
- Three fixed seeds and five folds completed.
- All five emotion labels completed for all representations.

## Registered Criteria

### Emotion-specific reference-frame heterogeneity

Supported in all five corpora. Range across emotions of Relative-minus-Absolute speaker-balanced F1:
- ESD: 0.1361.
- MEAD: 0.1828.
- RAVDESS: 0.1409.
- MSP: 0.4503.
- MELD: 0.1109.

### MSP class-localized baseline explanation

Supported.

Relative+Baseline minus Relative:
- neutral: +0.2685, 95% CI [0.2594, 0.2789].
- surprise: +0.0306 [0.0250, 0.0360].
- sad: -0.0262.
- angry: -0.0300.
- happy: -0.0772.

Thus at least two MSP emotions satisfy the preregistered >=0.02 improvement with CI above zero.

The MSP Absolute-over-Relative corpus-level reversal is overwhelmingly driven by neutral:
- neutral Relative-minus-Absolute: -0.3229, CI [-0.3301, -0.3166].
- angry: +0.0177.
- happy: +0.0539.
- sad: +0.1273.
- surprise: +0.0060.

### Conflict-set relevance

Supported.

Mean fraction of robust sign-conflict utterances:
- ESD: 30.1%.
- MEAD: 31.7%.
- RAVDESS: 29.1%.
- MSP: 19.5%.
- MELD: 20.3%.

Absolute-minus-Relative conflict-set accuracy:
- ESD: -0.2136.
- MEAD: -0.1912.
- RAVDESS: -0.1288.
- MSP: +0.0736.
- MELD: +0.1279.

Thus conflict examples are common, and the representation favored on those examples flips between controlled and natural corpora.

## Verdict

- Validity: valid.
- Decision: pass.
- Emotion-specific heterogeneity: supported.
- MSP class-localized baseline mechanism: supported.
- Conflict-set relevance: supported.
