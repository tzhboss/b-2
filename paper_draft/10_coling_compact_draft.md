# Prosodic Reference Frames: When Should Speaker Normalization Help Emotion Recognition?

## Abstract

Speaker normalization is widely used in speech emotion recognition (SER) to suppress
speaker-dependent variation. However, stable speaker information is not necessarily nuisance:
population-level affect targets can themselves contain between-speaker structure. We formulate
prosodic normalization as a reference-frame choice by decomposing acoustic features and continuous
affect targets into between-speaker baselines and within-speaker deviations. We then introduce a
controlled target intervention that continuously changes the strength of the target's
between-speaker component while keeping utterances, acoustic inputs, splits, and model family
fixed. Using semantically matched Pitch + Speaking Rate features, Relative-minus-Absolute CCC
decreases significantly with between-speaker target strength for Arousal and Dominance in both
MSP-Podcast and IEMOCAP under linear and quadratic Ridge models. At the pure within-speaker
endpoint, Relative outperforms Absolute in all confirmatory cells. A speaker-center permutation
intervention further shows that raw MSP performance requires the center of the correct speaker,
while this effect nearly disappears after within-speaker target centering. Label-free enrollment
experiments show that roughly 20 utterances recover most practical center utility on MSP. These
results support a conditional principle: representation reference frame should match target
reference frame.

## 1 Introduction

Speech simultaneously carries linguistic content, affective state, and stable characteristics of
the speaker. These sources of variation overlap acoustically. In SER, stable inter-speaker
differences are commonly treated as nuisance: systems normalize speaker statistics, learn
speaker-invariant representations, or adapt feature distributions across speakers. This choice is
reasonable when the goal is to recognize the same affective state independent of a person's
habitual voice.

Yet the target is not always speaker-invariant. Population-level dimensional affect labels may
contain stable differences across speakers, while other applications explicitly care about change
relative to a person's own baseline. These two targets ask different questions:

- How affective is this utterance compared with the population?
- How far is this speaker from their own typical state?

A representation that removes stable speaker information can be well matched to the second target
and poorly matched to the first.

We formalize this distinction with two decompositions. For acoustic prosody,

x_(s,u) = mu_s + delta_(s,u),

where mu_s is the speaker baseline and delta_(s,u) is within-speaker deviation. For a continuous
affect target,

y_(s,u) = ybar_s + epsilon_(s,u),

where ybar_s captures between-speaker target structure.

This yields three natural acoustic reference frames:

- Absolute: x_(s,u)
- Relative: x_(s,u) - mu_s
- Hybrid: [x_(s,u)-mu_s, mu_s]

and one central prediction:

> As a target moves from within-speaker deviation toward population-level absolute affect,
> Relative should become less favorable relative to Absolute, while speaker-center information can
> become useful.

The key methodological problem is that ordinary task comparisons confound target structure with
datasets, labels, and models. We therefore construct a target intervention that changes only the
between-speaker component of the target.

Our main contributions are:

1. A reference-frame formulation that decomposes both prosodic representations and affect targets
   into between- and within-speaker components.
2. A controlled target intervention showing that Relative-versus-Absolute utility changes
   systematically with between-speaker target strength.
3. Semantically matched cross-corpus confirmation on MSP-Podcast and IEMOCAP using Pitch +
   Speaking Rate under linear and quadratic models.
4. A speaker-center permutation intervention demonstrating that correct center identity drives
   raw-target baseline utility.
5. A label-free K-shot analysis showing that useful speaker references can be estimated in a
   deployment-style setting.

## 2 Related Work

### Speaker normalization

Speaker-normalized affect recognition predates modern neural SER. Bone et al. (2012) used pitch,
intensity, and voice-quality features relative to a speaker neutral-state model for unsupervised
cross-corpus arousal rating. Busso et al. (2013) proposed Iterative Feature Normalization, reducing
inter-speaker differences using speaker-specific neutral speech while aiming to preserve emotional
variation. Mariooryad and Busso (2014) similarly compensated speaker and lexical variability.

Modern methods retain the same nuisance-suppression motivation. Gat et al. (2022) adversarially
suppressed speaker characteristics in SSL-based SER, while Lu et al. (2024) treated speakers as
domains and learned speaker-invariant emotion representations.

Our contribution is therefore not speaker normalization itself. Likewise, person-mean centering is
a standard way to separate within- and between-person effects in multilevel affect research. The
novelty we investigate is the **coupling between these two choices**: we ask whether changing the
target reference frame changes which acoustic reference frame is useful.

### Speaker dependence and personalization

The complementary literature exploits individual differences. Sridhar et al. (2018) found stronger
speaker-dependent traits for Valence. Sridhar and Busso (2022) personalized Valence models using
acoustically similar speakers. Tran et al. (2023) combined pretrained speech encoders with
speaker-conditioned personalization, and Triantafyllopoulos and Schuller (2024) used minimal
speaker enrollment to improve personalized SER and individual-level fairness.

Recent evidence also shows that speaker representations can retain useful emotion structure:
Ulgen et al. (2024) identified emotion-related intra-speaker clusters in modern speaker embeddings.

These results suggest that speaker information is neither purely nuisance nor purely useful. We
test a specific determinant: the reference frame of the target.

### Between- and within-speaker prosody

Phonetic studies have long modeled between- and within-speaker prosodic variability separately.
For example, Quené (2008) and Jacewicz et al. (2010) quantify both sources of speaking-rate
variation. We extend this distinction to the affect target itself and manipulate its
between-speaker component directly.

## 3 Reference-Frame Intervention

### 3.1 Acoustic representation

For the primary cross-corpus experiment, we use Pitch and Speaking Rate because their semantics are
clear in both corpora.

Pitch is represented on a semitone/log scale and rate is log transformed. The diagnostic speaker
center is the speaker-wise marginal median.

For acoustic vector x and speaker center mu:

Absolute = x

Relative = x - mu

Hybrid = [x - mu, mu]

IEMOCAP loudness is excluded from the primary experiment because a source field used in preliminary
analyses does not agree with an RMS-dB reconstruction.

### 3.2 Target intervention

Let ybar_s be speaker s's target mean and ybar the global target mean. We define

y_lambda(s,u)
= ybar
+ [y_(s,u) - ybar_s]
+ lambda [ybar_s - ybar],

with lambda in {0, .25, .5, .75, 1}.

lambda=0 removes all between-speaker target offsets while preserving within-speaker variation.
lambda=1 exactly recovers the observed raw target.

The utterances, acoustic inputs, folds, and model family remain fixed across lambda.

### 3.3 Hypothesis

Define

Delta(lambda)
= CCC(Relative; lambda) - CCC(Absolute; lambda).

Reference-frame matching predicts a negative slope:

d Delta / d lambda < 0.

At lambda=0, Relative should be favored because the target is purely within-speaker. As
between-speaker structure increases, removing the speaker acoustic baseline should become less
appropriate.

## 4 Experimental Setup

### Continuous affect corpora

**MSP-Podcast.**
The main MSP analysis contains approximately 197k eligible utterances from about 1.9k speakers.
MSP provides continuous Valence, Arousal, and Dominance targets and sufficient speaker coverage for
mechanism and deployment experiments.

**IEMOCAP.**
IEMOCAP provides continuous dimensional annotations for 10 speakers. We use Pitch and Speaking Rate
only for the primary cross-corpus result.

### Categorical motivation

ESD, MEAD, and RAVDESS are used for an initial state-versus-trait comparison between five-class
emotion and gender classification.

### Models

The confirmatory intervention uses:
- Linear Ridge;
- Quadratic Ridge using degree-2 polynomial expansion.

No corpus-specific tuning is performed.

### Metrics and splits

Continuous affect is evaluated with speaker-balanced CCC. Categorical tasks use Macro-F1.
Primary experiments use speaker-disjoint evaluation.

## 5 Results

### 5.1 Categorical motivation: the same normalization helps state and hurts trait

Pitch Relative-minus-Absolute Macro-F1 for emotion is positive across all three categorical
corpora:

- ESD: +0.091
- MEAD: +0.090
- RAVDESS: +0.059

For gender, the same transformation is strongly negative:

- ESD: -0.133
- MEAD: -0.379
- RAVDESS: -0.230

This establishes that normalization redistributes task-relevant information rather than simply
improving every task.

### 5.2 MSP raw targets contain stable speaker structure

MSP between-speaker variance fractions are approximately:
- Valence: 0.21
- Arousal: 0.42
- Dominance: 0.35

Stable all-prosody speaker baselines alone predict speaker-level means with CCC 0.482 for Arousal
and 0.441 for Dominance, but only 0.022 for Valence.

After centering the target within speaker, Relative-minus-Absolute CCC becomes:
- Arousal: +0.131
- Dominance: +0.092

Adding the speaker baseline back contributes less than 0.001 CCC.

### 5.3 Primary result: target reference frame controls representation preference

Using Pitch + Speaking Rate, all eight confirmatory Arousal/Dominance slopes are negative and their
95% CIs are below zero.

Linear Ridge:

| Corpus | Arousal slope | Dominance slope |
|---|---:|---:|
| MSP | -0.318 | -0.182 |
| IEMOCAP | -0.022 | -0.011 |

Quadratic Ridge:

| Corpus | Arousal slope | Dominance slope |
|---|---:|---:|
| MSP | -0.321 | -0.185 |
| IEMOCAP | -0.026 | -0.018 |

At lambda=0, Relative-minus-Absolute CCC is positive in all eight corpus-by-target-by-model cells,
ranging from +0.046 to +0.155.

The effect is much steeper on MSP, consistent with its stronger between-speaker target structure.

### 5.4 Correct speaker-center identity explains raw baseline utility

We keep Relative features fixed but replace the appended speaker center with the center of another
speaker.

On raw MSP:
- Arousal true-center minus permuted-center: +0.296 CCC
- Dominance: +0.236 CCC

On within-speaker residual targets:
- Arousal: +0.001
- Dominance: +0.001

Thus extra center-valued dimensions are not sufficient: useful baseline information must correspond
to the correct speaker and to target structure that still contains speaker-level information.

### 5.5 Unlabeled enrollment recovers most practical center utility

On a fixed MSP downstream pool, K-shot Hybrid approaches the marginal-speaker oracle as K
increases.

At K=20:
- Arousal CCC = 0.490 versus oracle 0.495
- Dominance CCC = 0.384 versus oracle 0.384

The K20-to-K50 gain is only +0.004 Arousal CCC and approximately zero for Dominance.

This shows that the mechanism need not rely on labeled neutral enrollment.

## 6 Discussion

Our results suggest that the common question "Should speaker information be normalized away?" is
underspecified.

If the target represents deviation from a person's own typical state, stable speaker information is
nuisance by construction, and Relative prosody is well matched to the target.

If the target retains stable between-speaker differences, the speaker baseline can itself be
predictive signal. In that regime, removing it may hurt while explicitly exposing it may help.

This view reconciles speaker-invariant and personalized SER. They are not necessarily competing
philosophies; they correspond to different assumptions about target reference frame.

The closest historical work already shows that speaker-relative prosody can improve arousal
prediction. The present contribution is not the normalization procedure itself, but the controlled
demonstration that the utility of that procedure changes when the target's between-speaker
component is manipulated.

### Limitations

The continuous cross-corpus evidence is limited to MSP and IEMOCAP. IEMOCAP contains only ten
speakers and five dyadic sessions. A stricter session-disjoint sensitivity analysis shows
substantial session heterogeneity, so we do not claim session-level universality.

IEMOCAP loudness is excluded from the main analysis because feature provenance is ambiguous.
Valence remains weak for the low-dimensional prosody mechanism. Speaker identity is only modestly
suppressed by Relative low-dimensional prosody, so gender should be interpreted as a strong trait
example rather than evidence for complete speaker invariance.

Finally, lambda is a controlled analysis variable rather than an observable property supplied by a
real deployment task. The practical implication is therefore to make the intended population-level
versus person-relative meaning of the target explicit.

## 7 Conclusion

Speaker normalization is not neutral. It changes which sources of variation remain available to
the model.

Across cross-corpus target intervention, center-identity permutation, and label-free enrollment, the
evidence supports a conditional principle:

**Representation reference frame should match target reference frame.**

SER systems should therefore specify whether they aim to predict population-level affect or
within-person deviation and choose their acoustic representation accordingly.
