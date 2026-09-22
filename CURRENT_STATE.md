# CURRENT_STATE.md

## Phase

WavLM embedding-level target-reference intervention completed and passed.

## New strongest extension

On IEMOCAP frozen WavLM embeddings:
- layer 12 Arousal/Dominance slopes are significantly negative;
- layer 24 Arousal/Dominance slopes are significantly negative;
- all four pure within-speaker endpoints significantly favor Relative;
- Arousal slope is negative for all 10 speakers at both layers.

Therefore target-reference matching is not limited to handcrafted prosodic level features.

## Statistical status

- MSP explicit Pitch+Rate: strong speaker-cluster confirmation.
- IEMOCAP explicit Pitch+Rate: directionally consistent slopes but low-powered slope CIs.
- IEMOCAP WavLM: strong speaker-consistent and speaker-cluster target-reference effect.

## Next legal step

Run a fixed numerical-solver sensitivity for the WavLM Ridge probe because the default solver
reported ill-conditioned matrix warnings. Then update the paper evidence hierarchy and figures.
