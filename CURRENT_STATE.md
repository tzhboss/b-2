# CURRENT_STATE.md

## Phase

Unseen-speaker K-shot pitch-reference experiment registered; execution pending.

## Active research question

How much label-free reference audio is required to estimate a useful speaker pitch baseline for
emotion classification on entirely unseen speakers?

## Valid evidence

- EXP-20260920-02: explicit prosody-only reference-frame interaction supported.
- EXP-20260920-03: final-layer WavLM conditional Absolute-vs-Relative effect inconclusive.
- EXP-20260920-04: layer-dependent pitch structure supported; middle-layer downstream increment inconclusive.

## Next legal step

Run EXP-20260920-05 with speaker-disjoint folds and K=[1,2,5,10] label-free enrollment,
audit speaker/enrollment leakage, and compare K-shot relative pitch against matched Absolute and
Oracle-relative upper bounds.
