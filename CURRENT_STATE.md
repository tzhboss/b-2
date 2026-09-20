# CURRENT_STATE.md

This file is the canonical short current-state summary. Keep it approximately 50–100 lines or less and do not use it as a historical ledger.

## Phase

First corrected controlled prosody reference-frame pilot completed, audited, and promoted.

## Active research question

Does the useful reference frame for explicit prosody depend on both downstream task
(emotion state versus gender trait) and acoustic attribute (pitch, loudness, rate)?

## Valid evidence

- EXP-20260920-02: valid, pass. Audited results are in results/EXP-20260920-02/.
- Source evidence is a pinned legacy enriched snapshot, not the unfinished current English-final release.
- Evidence is limited to reference-available / known-speaker controlled-corpus probes.

## Recent key experiments

- EXP-20260920-01 — completed, invalid, inconclusive. Superseded due to a RAVDESS fold-level class-coverage / macro-F1 label-universe issue.
- EXP-20260920-02 — completed, valid, pass. Four-fold correction with full-label validation.

## Main audited observation

Pitch shows a replicated task reversal across ESD-English, MEAD-part0, and RAVDESS-speech:
speaker-relative pitch improves emotion macro-F1, while absolute pitch preserves substantially
more gender-predictive information. Loudness and rate show different patterns, supporting an
attribute-dependent rather than universal reference-frame effect.

## Blocker

Broader claims require uncontrolled-corpus and unseen-speaker/reference-estimation experiments.

## Next legal step

Register a follow-up experiment for uncontrolled corpora and/or unseen-speaker reference
estimation without modifying the completed EXP-20260920-02 hypothesis.
