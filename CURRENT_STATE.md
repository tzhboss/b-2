# CURRENT_STATE.md

## Phase

MSP raw-VAD K-shot deployment experiment completed and audited.

## Main valid VAD findings

- Raw Arousal/Dominance reward stable speaker-level prosodic information.
- Within-speaker VAD residuals strongly prefer Relative pitch/loudness/all-prosody.
- These mechanisms survive low human-label disagreement.
- Label-free K-shot acoustic enrollment converges toward a speaker marginal acoustic center.
- K=10 Hybrid is already within 0.006 CCC of the marginal-oracle Hybrid for Arousal and within
  0.001 CCC for Dominance.

## Next legal step

Measure K-shot sample-efficiency and saturation beyond K=10 using preregistered K=20 and K=50
without changing the estimator or model.
