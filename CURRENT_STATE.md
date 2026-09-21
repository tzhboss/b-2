# CURRENT_STATE.md

## Phase

Speaker-cluster inference completed and audited; WavLM embedding intervention extraction completed.

## Statistical conclusion

- MSP target-reference slopes remain strongly negative under speaker-cluster bootstrap.
- IEMOCAP slope means remain negative, but 10-speaker cluster-bootstrap CIs cross zero.
- Pure within-speaker Relative advantage is robust in both corpora.

Main paper inference should use EXP-20260921-15 rather than fold/seed CIs.

## Next legal step

Run EXP-20260921-16 WavLM embedding-level target-reference probe using completed layer-12/layer-24
IEMOCAP embeddings.
