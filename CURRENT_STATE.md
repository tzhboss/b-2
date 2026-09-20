# CURRENT_STATE.md

## Phase

Prosody-only and frozen-WavLM reference-frame pilots completed and audited.

## Active research question

How does prosodic reference-frame dependence change between explicit acoustic representations
and learned speech representations?

## Valid evidence

- EXP-20260920-02: valid, pass. Prosody-only pitch shows a replicated task reversal across ESD,
  MEAD, and RAVDESS: speaker-relative pitch improves emotion probing while absolute pitch retains
  substantially more gender-predictive information.
- EXP-20260920-03: valid, inconclusive under its preregistered WavLM conditional-reversal
  criterion. Final-layer frozen WavLM-large largely collapses the incremental difference between
  explicit Absolute and Relative pitch.
- EXP-20260920-03 additionally shows statistically resolved gains from Hybrid/all-prosody explicit
  cues over WavLM-only for ESD emotion and smaller gains for MEAD emotion.

## Recent key experiments

- EXP-20260920-01 — completed, invalid, inconclusive.
- EXP-20260920-02 — completed, valid, pass.
- EXP-20260920-03 — completed, valid, inconclusive.

## Main current interpretation

Reference-frame dependence appears representation-dependent. The strong effect in explicit
prosody-only probes is largely absent after conditioning on final-layer mean-pooled WavLM-large,
although combined explicit prosody can still add emotion-relevant information.

## Blocker

The current WavLM result uses only the final layer, known-speaker folds, and a near-saturated
gender probe. Broader claims require layer-wise analysis, less saturated trait tasks, and
unseen-speaker/reference-estimation evaluation.

## Next legal step

Register a layer-wise WavLM probe and a less saturated trait target such as speaker identity or
speaker-baseline prediction before changing the main paper claim.
