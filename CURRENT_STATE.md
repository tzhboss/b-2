# CURRENT_STATE.md

## Phase

Core experiments are consolidated into paper-ready evidence.

## Strongest confirmatory evidence

- Categorical motivation: EXP-20260920-02.
- MSP target decomposition: EXP-20260920-17.
- Semantically matched MSP/IEMOCAP Pitch+Rate target intervention: EXP-20260921-12.
- Correct-speaker center identity intervention: EXP-20260921-06.
- Fixed-pool K-shot deployment: EXP-20260920-21.
- WavLM reference-frame decodability: EXP-20260920-04.

## Key limitations already identified

- IEMOCAP loudness field relative_db is semantically unreliable for the main claim.
- Strict IEMOCAP session-disjoint slope replication is heterogeneous across only five sessions;
  EXP-20260921-13 is invalid for its original CI due duplicated deterministic seeds and is retained
  only as corrected session-level sensitivity.
- Speaker identity is only modestly suppressed by Relative low-dimensional prosody.
- Valence remains weak for the low-dimensional prosody mechanism.

## Paper framing

Do not frame the contribution as new speaker normalization.
Frame it as a reference-frame matching principle:
speaker baseline is nuisance or useful information depending on the between/within-speaker
structure of the target.

## Next legal step

Complete a focused literature-gap review and draft the Introduction/Related Work/Methods around the
confirmed mechanism, while avoiding unsupported novelty claims.
