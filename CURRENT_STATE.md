# CURRENT_STATE.md

## Phase

Continuous MSP VAD reference-frame experiment completed and audited.

## Main valid findings

- Continuous human-rated MSP Arousal and Dominance strongly prefer Absolute over Relative prosody.
- Adding stable speaker baseline back to Relative recovers large CCC gains, especially for
  loudness and all-attribute representations.
- Valence is essentially not predicted by pitch/loudness/rate alone under any reference frame.
- The VAD target itself likely mixes between-speaker affective priors with within-speaker state.

## Next legal step

Decompose each VAD target into speaker mean plus within-speaker residual, then test whether stable
prosodic baselines predict between-speaker VAD means while Relative prosody better predicts
within-speaker VAD deviations.
