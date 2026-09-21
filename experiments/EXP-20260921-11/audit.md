# Experiment Audit — EXP-20260921-11

## Runtime integrity and split audit

- Source/runtime script and config SHA256 values match.
- Immutable /tmp execution used.
- All content-disjoint folds are usable:
  - ESD: 15/15 per attribute set.
  - MEAD: 15/15.
  - RAVDESS: 12/12.
- No content group appears in both train and test within a fold.
- Every speaker class is represented in train and test for every retained fold.
- No metric NaNs.

## Speaker identity decodability

All-prosody Macro-F1:

ESD:
- Absolute: 0.1787.
- Relative: 0.1612.
- Baseline-only: 0.6000.
- Relative+Baseline: 0.7450.

MEAD:
- Absolute: 0.0674.
- Relative: 0.0327.
- Baseline-only: 0.0815.
- Relative+Baseline: 0.2606.

RAVDESS:
- Absolute: 0.0882.
- Relative: 0.0657.
- Baseline-only: 0.2290.
- Relative+Baseline: 0.4461.

Relative-minus-Absolute all-prosody:
- ESD: -0.0176, 95% CI [-0.0194, -0.0155].
- MEAD: -0.0347, [-0.0370, -0.0322].
- RAVDESS: -0.0225, [-0.0282, -0.0167].

Pitch-only suppression is much smaller:
- ESD: -0.0213.
- MEAD: -0.00037.
- RAVDESS: -0.00222.

## Registered criteria

- Strong speaker-identity suppression: not supported.
- Baseline-only identity concentration at the preregistered >=0.20 margin in at least two corpora:
  not supported.
- Hybrid restoration: supported.

## Interpretation

Speaker-relative normalization does reduce speaker-identity decodability, but the effect is modest
for these low-dimensional prosodic summaries and is much weaker than the previously observed
gender reversal.

Explicit baseline information is strongly speaker-specific in ESD and RAVDESS and, when combined
with Relative features, substantially increases speaker identification in all three corpora.

This experiment therefore constrains the paper claim: gender should be treated as a strong
speaker-trait example, not as evidence that Relative prosody universally removes speaker identity.

## Verdict

- Validity: valid.
- Decision: mixed.
