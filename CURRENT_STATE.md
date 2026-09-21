# CURRENT_STATE.md

## Phase

Semantically safe IEMOCAP pitch/rate external validation completed and audited.

## Main finding

IEMOCAP target-reference coupling survives after removing all loudness fields. Pitch+rate shows
significantly negative Arousal/Dominance Relative-minus-Absolute slopes under both linear and
quadratic Ridge, with positive Relative advantages at the pure within-speaker endpoint.

## Evidence boundary

- IEMOCAP pitch/rate: main external evidence.
- IEMOCAP loudness: excluded from main claims because relative_db semantics are undocumented and
  RMS-dB sensitivity reverses the result.
- MSP pitch/loudness/rate: remains the large-speaker primary corpus.

## Next legal step

Build a paper-ready evidence synthesis across valid experiments: claim matrix, main effect table,
boundary/failure table, and figure-source CSVs for the target-reference matching story.
