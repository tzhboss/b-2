# CURRENT_STATE.md

## Phase

Deployment-realistic K-shot raw-VAD experiment completed and audited.

## Main valid VAD findings

- Raw Arousal/Dominance reward stable between-speaker prosodic information.
- Within-speaker VAD residuals strongly prefer Relative prosody.
- These mechanisms survive low human-label disagreement.
- K=10 unlabeled acoustic enrollment reduces center-estimation error by roughly 68-71%.
- K=10 Hybrid significantly improves raw Arousal and Dominance over Absolute and is very close to
  the diagnostic marginal-speaker oracle.
- Relative without exposing the estimated baseline remains strongly inappropriate for raw VAD.

## Next legal step

Measure K-shot sample-efficiency and saturation on a fixed high-support speaker cohort using
K=[1,2,5,10,20,50], keeping the estimator/model fixed, to determine whether K=10 is near the
practical plateau.
