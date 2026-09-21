# 4. Results

## 4.1 Speaker-relative pitch reveals a state-versus-trait reversal

We first ask whether speaker-relative prosody redistributes information between an affective state
task and a stable speaker-trait task. Using pitch alone, Relative-minus-Absolute Macro-F1 is
positive for five-class emotion recognition and strongly negative for gender classification in all
three categorical corpora.

For emotion, Relative improves Macro-F1 by:

- ESD: +0.0908, 95% CI [0.0870, 0.0947]
- MEAD: +0.0904, 95% CI [0.0876, 0.0933]
- RAVDESS: +0.0586, 95% CI [0.0417, 0.0724]

For gender, the same transformation changes Macro-F1 by:

- ESD: -0.1330, 95% CI [-0.1390, -0.1265]
- MEAD: -0.3790, 95% CI [-0.3817, -0.3764]
- RAVDESS: -0.2299, 95% CI [-0.2427, -0.2158]

This reversal motivates the reference-frame hypothesis but does not by itself explain why the
preferred representation changes.

## 4.2 MSP target decomposition identifies substantial speaker-level Arousal/Dominance structure

We next decompose MSP continuous targets into between-speaker and within-speaker components.

The ICC-like between-speaker variance fractions are:

- Valence: 0.213
- Arousal: 0.418
- Dominance: 0.346

Thus Arousal and Dominance contain substantially more stable speaker-level target structure than
Valence.

Consistent with this observation, stable all-prosody speaker baselines alone predict the
speaker-level target mean with CCC:

- Arousal: 0.482
- Dominance: 0.441
- Valence: 0.022

The effect reverses once the target is centered within speaker. On within-speaker residual targets,
all-prosody Relative-minus-Absolute CCC is:

- Arousal: +0.1306, 95% CI [0.1282, 0.1326]
- Dominance: +0.0921, 95% CI [0.0894, 0.0945]
- Valence: +0.00495, 95% CI [0.00466, 0.00527]

Adding the stable speaker baseline back to Relative features contributes almost nothing after target
centering:

- Arousal: +0.00049 CCC
- Dominance: +0.00029 CCC
- Valence: +0.00002 CCC

These results support the decomposition account: baseline information is useful when the target
retains speaker-level structure and becomes largely irrelevant after that structure is removed.

## 4.3 Controlled target intervention confirms reference-frame coupling

The central experiment manipulates target reference frame directly rather than comparing unrelated
tasks. We define lambda from 0 to 1, where lambda=0 removes the between-speaker target component
and lambda=1 recovers the original raw target.

To avoid feature-semantic mismatch across corpora, the confirmatory experiment uses only Pitch +
Speaking Rate in both MSP-Podcast and IEMOCAP.

The primary statistic is the slope of

CCC(Relative) - CCC(Absolute)

as a function of lambda.

### Linear Ridge

- MSP Arousal: -0.3181, 95% CI [-0.3264, -0.3082]
- MSP Dominance: -0.1821, 95% CI [-0.1878, -0.1750]
- IEMOCAP Arousal: -0.02245, 95% CI [-0.03772, -0.00838]
- IEMOCAP Dominance: -0.01089, 95% CI [-0.01684, -0.00348]

### Quadratic Ridge

- MSP Arousal: -0.3207, 95% CI [-0.3293, -0.3103]
- MSP Dominance: -0.1846, 95% CI [-0.1906, -0.1771]
- IEMOCAP Arousal: -0.02646, 95% CI [-0.04310, -0.01134]
- IEMOCAP Dominance: -0.01752, 95% CI [-0.02440, -0.00994]

All eight confirmatory slopes are significantly negative.

At the pure within-speaker endpoint lambda=0, Relative-minus-Absolute CCC is positive in every
confirmatory cell:

- MSP Linear: +0.1389 Arousal, +0.0727 Dominance
- MSP Quadratic: +0.1389 Arousal, +0.0746 Dominance
- IEMOCAP Linear: +0.1363 Arousal, +0.0458 Dominance
- IEMOCAP Quadratic: +0.1551 Arousal, +0.0486 Dominance

The direction is therefore stable across corpus and model family: increasing between-speaker target
structure makes Relative progressively less favorable relative to Absolute.

## 4.4 Correct speaker-center identity is necessary for raw MSP baseline utility

The preceding results show coupling between target structure and reference-frame utility. We next
ask whether Hybrid gains actually require the center to belong to the correct speaker.

We preserve each utterance's Relative features but randomly reassign center vectors across speakers,
using derangements that preserve the center marginal distribution.

For raw MSP targets, true-center Hybrid exceeds mean permuted-center Hybrid by:

- Arousal: +0.2955 CCC, 95% CI [0.2856, 0.3039]
- Dominance: +0.2363 CCC, 95% CI [0.2285, 0.2436]
- Valence: +0.0052 CCC

After target centering within speaker, the same true-minus-permuted contrast collapses to:

- Arousal: +0.00139
- Dominance: +0.00100
- Valence: +0.00006

Thus the raw-target benefit is not explained by adding arbitrary center-valued dimensions. It
depends on alignment between the correct speaker center and the speaker-level component of the
target.

IEMOCAP shows the same qualitative boundary at much smaller magnitude, consistent with its weaker
between-speaker VAD structure.

## 4.5 Label-free enrollment recovers practical speaker-reference utility

Oracle speaker centers require access to a speaker's full history. We therefore estimate acoustic
centers from K unlabeled enrollment utterances.

The corrected MSP analysis reserves the same enrollment pool and keeps downstream rows fixed across
all K.

Arousal K-shot Hybrid CCC increases from:

- K=1: 0.4795
- K=10: 0.4871
- K=20: 0.4903
- K=50: 0.4946

The marginal-oracle Hybrid CCC is 0.4946.

Dominance K-shot Hybrid CCC is:

- K=1: 0.3793
- K=10: 0.3837
- K=20: 0.3839
- K=50: 0.3840

The marginal-oracle Hybrid CCC is 0.3835.

At K=20, the gap to the marginal oracle is only -0.0043 CCC for Arousal and approximately zero for
Dominance. Increasing K from 20 to 50 yields +0.0043 CCC for Arousal and +0.0001 for Dominance.

This supports practical, label-free reference estimation, while not implying a universal optimal K
across corpora.

## 4.6 WavLM retains information corresponding to multiple reference frames

We probe frozen WavLM-large layers for three pitch targets: Absolute pitch, Relative pitch, and the
implied speaker baseline.

All three are strongly decodable.

Peak cross-validated R² values are:

### ESD
- Absolute pitch: 0.958 at layer 3
- Relative pitch: 0.895 at layer 3
- Implied speaker baseline: 0.983 at layer 6

### MEAD
- Absolute pitch: 0.987 at layer 3
- Relative pitch: 0.861 at layer 4
- Implied speaker baseline: 0.971 at layer 4

### RAVDESS
- Absolute pitch: 0.939 at layer 9
- Relative pitch: 0.892 at layer 5
- Implied speaker baseline: 0.979 at layer 4

The result shows that a modern SSL encoder retains information associated with multiple prosodic
reference frames. However, explicit Relative-versus-Absolute emotion gains after WavLM are small
and unresolved, so the result should be interpreted as decodability rather than evidence that
Relative augmentation universally improves SSL-based SER.

## 4.7 Speaker identity is reduced only modestly by low-dimensional Relative prosody

Gender decoding shows a strong trait-information reversal, but speaker identity provides a more
conservative test.

Content-disjoint all-prosody speaker-ID Macro-F1 is:

### ESD
- Absolute: 0.179
- Relative: 0.161
- Baseline-only: 0.600
- Relative+Baseline: 0.745

### MEAD
- Absolute: 0.067
- Relative: 0.033
- Baseline-only: 0.081
- Relative+Baseline: 0.261

### RAVDESS
- Absolute: 0.088
- Relative: 0.066
- Baseline-only: 0.229
- Relative+Baseline: 0.446

Relative reduces speaker-ID decodability, but the reduction is modest compared with the gender
effect. We therefore avoid claiming that low-dimensional speaker normalization removes speaker
identity.

## 4.8 Negative and sensitivity results constrain the claim

Several analyses provide important boundaries.

First, IEMOCAP's source field relative_db is not a reliable basis for a loudness claim. Replacing
it with physically interpretable RMS dB removes or reverses the target-reference slope. Main
cross-corpus claims therefore exclude IEMOCAP loudness.

Second, a strict IEMOCAP leave-one-session-out sensitivity analysis reveals substantial
session-level heterogeneity. The original implementation also repeated the same deterministic five
session folds under three seed labels; its inferential CI was therefore invalid and was discarded.
After collapsing to the five unique sessions, mean Arousal/Dominance slopes remain negative, but
only two to three of five sessions show negative slopes in each model-target cell.

Third, K-shot mechanism recovery on IEMOCAP is not monotonic in K despite improved physical center
estimation. Better reference estimation does not guarantee monotonic recovery of a downstream
mechanism when only ten speakers are available.

Together these results motivate a scoped conclusion: target-reference matching is strongly
supported in the main speaker-disjoint analyses, while corpus/session composition and feature
semantics materially affect effect magnitude.
