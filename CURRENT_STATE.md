# CURRENT_STATE.md

## Phase

Fixed-pool K-shot MSP VAD saturation completed and audited.

## Main valid VAD conclusions

- Raw population-level Arousal/Dominance reward stable speaker-level prosodic information.
- Within-speaker VAD residuals strongly prefer speaker-relative pitch/loudness/all-prosody.
- Both mechanisms survive low annotator disagreement.
- Label-free unseen-speaker acoustic enrollment can recover raw-VAD Hybrid utility.
- On a fixed downstream pool, K=20 is practically saturated near the marginal-speaker oracle;
  K=50 provides only a small additional Arousal gain and negligible Dominance gain.

## Next legal step

Seek a second human-rated dimensional-affect corpus, prioritizing IEMOCAP, and reproduce the raw
VAD versus within-speaker residual decomposition with matched prosodic reference frames.
