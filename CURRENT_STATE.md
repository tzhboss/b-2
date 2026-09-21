# CURRENT_STATE.md

## Phase

Extended attribute experiment is running; corrected speaker-cluster inference is registered.

## Statistical correction

EXP-20260921-15 will generate one OOF prediction per utterance under a fixed speaker-disjoint split
and compute uncertainty by resampling complete speakers, not seed/fold aggregate rows.

## Next legal step

Finish EXP-20260921-14, then run EXP-20260921-15 from an immutable runtime bundle.
