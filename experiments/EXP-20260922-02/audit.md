# Experiment Audit — EXP-20260922-02

## Runtime integrity

- Completed with exit code 0 on 2026-09-22.
- Immutable runtime bundle script/config hashes match the preregistered source hashes.
- Preregistered runtime commit: 76e2e80.
- Fixed 50-utterance unlabeled reserve per speaker with nested K prefixes.
- K values: 1, 2, 5, 10, 20, 50.
- Three seeds and five speaker-disjoint folds; downstream rows are fixed across K within seed.
- Six interpretable center dimensions: pitch, F0 variability, loudness, speaking rate, pause ratio, voiced ratio.

## Registered acceptance checks

### K=20 Hybrid exceeds Absolute

- Arousal: +0.039015 CCC, 95% CI [+0.032989, +0.045401].
- Dominance: +0.031775 CCC, 95% CI [+0.027152, +0.036296].

Both targets exceed Absolute: supported.

### K=20 to K=50 plateau

- Arousal K50-K20: +0.009295 CCC, 95% CI [+0.006828, +0.011660].
- Dominance K50-K20: +0.007391 CCC, 95% CI [+0.005442, +0.009142].

Both mean gains are below the preregistered 0.01 threshold: supported.

### K=50 oracle proximity

- Arousal K=50 K-shot Hybrid: 0.568000; marginal-oracle Hybrid: 0.572431; gap 0.004431.
- Dominance K=50 K-shot Hybrid: 0.464414; marginal-oracle Hybrid: 0.468073; gap 0.003659.

Both gaps are below 0.01 CCC: supported.

## Additional observations

Center-estimation MAE decreases from K=1 to K=50 for all six attributes. K=20 is already useful but is not fully oracle-like: its oracle gaps are 0.013726 CCC for Arousal and 0.011050 for Dominance. K=50 closes most of the remaining gap.

The Arousal K20-to-K50 mean gain (0.009295) passes the preregistered <0.01 plateau threshold narrowly, while its confidence interval extends above 0.01. The preregistered criterion was defined on the mean gain, so the acceptance decision remains pass; the paper should avoid claiming a sharp statistical plateau at exactly K=20.

## Verdict

- Validity: valid.
- Decision: pass.
