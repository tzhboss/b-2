# Experiment Audit — EXP-20260920-10

## Completeness

- Per-emotion fold rows: 3,000 / 3,000 expected.
- Per-emotion summary rows: 200 / 200 expected.
- Per-emotion paired-delta rows: 150 / 150 expected.
- Conflict fold rows: 150 / 150 expected.
- Conflict summary rows: 10 / 10 expected.
- Data-inventory rows: 10 / 10 expected.
- Per-emotion metric NaNs: zero.
- Speaker-baseline decomposition is constant within speaker to floating-point precision.

Some conflict-fold accuracies are undefined because the preregistered fixed magnitude thresholds
produce zero conflict samples in 17 fold-by-attribute cells. These cells remain NaN by design and
are not imputed or threshold-adjusted post hoc.

## Registered Criterion 1: Cross-attribute class heterogeneity

Supported.

Relative-minus-Absolute F1 range across the five emotions:

- ESD loudness: 0.0505.
- ESD rate: 0.0461.
- MEAD loudness: 0.0604.
- MEAD rate: 0.0068.
- MELD loudness: 0.0372.
- MELD rate: 0.0512.
- MSP loudness: 0.1381.
- MSP rate: 0.5940.
- RAVDESS loudness: 0.0393.
- RAVDESS rate: 0.0891.

Nine of ten corpus-by-attribute cells exceed the preregistered 0.03 threshold.

## Registered Criterion 2: Baseline redistribution

Supported.

Five corpus-by-attribute cells contain at least two emotions with absolute baseline-addition
effects >=0.02 and paired-bootstrap confidence intervals excluding zero:

- MELD loudness.
- MELD rate.
- MSP loudness.
- MSP rate.
- RAVDESS loudness.

The preregistered requirement was four cells.

## Registered Criterion 3: Broad conflict-set relevance

Not supported under the fixed preregistered magnitude thresholds.

Conflict fractions >=0.05 occur in only five of ten corpus-by-attribute cells, below the required
eight. Several rate/MELD cells have very sparse or empty conflict folds.

Where conflict support is adequate, large Absolute-versus-Relative accuracy differences still
occur, including:
- MEAD loudness: Absolute minus Relative = -0.0994.
- MSP loudness: +0.0617.
- MSP rate: +0.2227.
- RAVDESS rate: -0.2219.

Because coverage failed the registered criterion, these are descriptive rather than broad
confirmatory evidence.

## Key Attribute-Specific Findings

### Loudness

- MEAD strongly prefers Relative loudness for neutral (+0.0825 F1) and happy (+0.0560).
- MSP neutral also prefers Relative loudness (+0.1284), the opposite of MSP pitch.
- ESD angry prefers Absolute loudness (-0.0322 Relative-minus-Absolute).
- Baseline addition has class-specific sign flips, including strong positive MELD-neutral
  (+0.2363) and strong negative MSP-neutral (-0.1121).

### Speaking Rate

- MSP is highly heterogeneous:
  - angry: Relative +0.1597.
  - surprise: +0.0362.
  - happy: +0.0169.
  - sad: -0.1651.
  - neutral: -0.4343.
- Adding rate baseline back recovers neutral (+0.2385) and sad (+0.1307), but hurts angry and happy.
- MEAD rate has only weak Relative-vs-Absolute class differences, even though baseline addition
  can help angry (+0.0419).

## Audit Verdict

- Validity: valid.
- Decision: mixed.
- Cross-attribute per-emotion heterogeneity: supported.
- Baseline redistribution: supported.
- Broad fixed-threshold conflict-set generalization: inconclusive / not supported.
