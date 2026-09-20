# CURRENT_STATE.md

## Phase

MSP VAD disagreement-robustness experiment completed and audited.

## Main valid VAD findings

- Raw Arousal/Dominance contain large between-speaker components predictable from stable prosodic baselines.
- Within-speaker Arousal/Dominance residuals strongly prefer Relative pitch/loudness/all-prosody.
- Both effects persist among low-disagreement human SAM ratings.
- A universal monotonic degradation with annotation disagreement is not supported.

## Next legal step

Test deployment-realistic K-shot unlabeled acoustic enrollment for raw VAD: estimate each unseen
speaker's acoustic center from K reference utterances and determine whether K-shot Hybrid recovers
the oracle/marginal speaker-baseline benefit.
