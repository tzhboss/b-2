# Experiment Report — EXP-20260920-05

## Registered Hypothesis
Speaker-relative pitch should remain useful under unseen-speaker evaluation when speaker baseline is estimated from a small unlabeled enrollment set. Larger K should improve baseline quality and emotion performance.

## Main Result
The hypothesis is strongly supported.

Absolute-pitch emotion macro-F1:
- ESD: 0.2154.
- MEAD: 0.1316.
- RAVDESS: 0.1840.

Relative pitch with K=10 unlabeled enrollment:
- ESD: 0.3169.
- MEAD: 0.2238.
- RAVDESS: 0.2670.

Relative-K10 improvement over Absolute:
- ESD: +10.15 percentage points.
- MEAD: +9.22 points.
- RAVDESS: +8.31 points.

All paired-bootstrap confidence intervals exclude zero.

## Reference Budget
Relative-K1 macro-F1:
- ESD 0.2353; MEAD 0.2032; RAVDESS 0.2184.

Relative-K10 improves over K1 by:
- ESD +8.16 points.
- MEAD +2.06 points.
- RAVDESS +4.86 points.

All three confidence intervals exclude zero.

Baseline-estimation MAE also decreases strongly with K in every corpus.

## Hybrid
Hybrid Absolute+Relative is robust, especially at K=1, but K=10 Relative alone is already strong:
- ESD Hybrid-K10 0.3208.
- MEAD Hybrid-K10 0.2257.
- RAVDESS Hybrid-K10 0.2681.

## Supported Claim
Speaker-relative pitch does not require an oracle speaker baseline to be useful. Under strict speaker-disjoint evaluation, a baseline estimated from a small unlabeled enrollment set yields large emotion-classification gains over absolute pitch, and the gain increases as more reference speech is available.

## Important Boundary
The enrollment baseline here is a generic speaker median over randomly selected unlabeled utterances, not the dataset-specific neutral reference used in earlier derived labels. This is intentionally a deployment-oriented reference frame and should not be conflated with neutral-anchored normalization.

## Decision
pass.

## Next Step
Test whether K-shot relative pitch still adds information on top of frozen WavLM under the exact same speaker-disjoint enrollment/target protocol.
