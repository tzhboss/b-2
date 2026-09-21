# Prosodic Reference Frames: Draft Manuscript

# Abstract

Speaker normalization is commonly used in speech emotion recognition (SER) to suppress
speaker-dependent variability, implicitly treating stable speaker characteristics as nuisance.
We argue that this assumption is task-dependent: whether speaker baseline information should be
removed depends on the reference frame of the prediction target.

We formulate prosodic reference frames by decomposing acoustic features into stable speaker
baselines and within-speaker deviations, and decompose continuous affect targets analogously into
between-speaker and within-speaker components. We then introduce a controlled target intervention
that continuously varies the strength of between-speaker target structure while keeping acoustic
inputs, train/test splits, and model family fixed.

Using semantically matched Pitch + Speaking Rate features, we find that
Relative-minus-Absolute CCC decreases significantly as between-speaker Arousal/Dominance target
structure increases in both MSP-Podcast and IEMOCAP, under both linear and quadratic Ridge models.
At the pure within-speaker endpoint, Relative outperforms Absolute in every confirmatory
corpus-by-target-by-model cell. A speaker-center permutation intervention further shows that the
large Hybrid benefit on raw MSP Arousal/Dominance depends on assigning the correct center to the
correct speaker, whereas the effect nearly disappears after within-speaker target centering.
Finally, label-free enrollment experiments show that roughly 20 utterances recover most of the
practical speaker-reference utility on a fixed MSP downstream pool.

These results support a conditional view of normalization: stable speaker information can be
nuisance or useful signal depending on target structure. We therefore propose that representation
reference frame should be matched to target reference frame rather than assuming that speaker
invariance is universally desirable.

# 1. Introduction

Speech emotion recognition (SER) models must separate affective variation from the many other
sources of acoustic variability carried by speech. Speaker identity is one of the most persistent
sources of such variation: speakers differ systematically in habitual pitch, loudness, speaking
rate, voice quality, accent, physiology, and expressive style. A common response is therefore to
normalize speaker-dependent variation or to learn speaker-invariant representations. Classical
feature-normalization approaches estimate speaker-specific reference statistics, often from neutral
speech, and transform expressive utterances relative to those references. More recent neural
approaches suppress speaker information adversarially or align speaker domains. Across these
lines of work, the usual premise is that speaker variability is primarily a nuisance that impairs
generalization.

That premise is useful, but incomplete. Emotion targets themselves can contain stable
between-speaker structure. Two speakers may differ not only in their acoustic baselines but also in
their population-level affect ratings, annotation distributions, or habitual expressive ranges. In
such a setting, removing all stable speaker information can discard signal that is predictive of the
target. Conversely, for a task that explicitly asks whether a speaker is above or below their own
typical state, stable speaker baselines should be nuisance by construction.

This distinction suggests that speaker normalization should be treated as a **reference-frame
choice**, not as a universally beneficial invariance operation. We write a simple acoustic
decomposition as

x_(s,u) = mu_s + delta_(s,u),

where mu_s is a stable speaker baseline and delta_(s,u) is an utterance-level deviation. A
speaker-relative representation suppresses mu_s and emphasizes delta_(s,u), whereas an absolute
representation retains both. A Hybrid representation can expose the deviation and baseline
separately. The target can be decomposed analogously:

y_(s,u) = ybar_s + epsilon_(s,u),

where ybar_s captures between-speaker target structure and epsilon_(s,u) captures within-speaker
variation. This leads to the central hypothesis of this work:

> **Representation reference frame should match target reference frame.**

The hypothesis predicts that speaker-relative prosody should be advantageous for within-speaker
targets, while absolute or Hybrid representations can become useful as the target contains more
between-speaker structure. Importantly, this is not a claim that speaker-relative prosody is always
better for emotion recognition.

We test this hypothesis through a sequence of controlled experiments. First, low-dimensional
prosody establishes an intuitive state-versus-trait reversal: speaker-relative pitch improves
categorical emotion classification while strongly reducing gender decoding across ESD, MEAD, and
RAVDESS. We then move to continuous valence-arousal-dominance (VAD) prediction. On MSP-Podcast,
raw Arousal and Dominance contain substantial between-speaker variance, and stable speaker acoustic
centers are strongly useful. Once the speaker mean is removed from the target, however,
speaker-relative prosody becomes preferable.

To move beyond post-hoc decomposition, we introduce a controlled target intervention. For each
speaker, we continuously vary the strength of the between-speaker target component while holding
utterances, acoustic inputs, folds, and model family fixed. With semantically aligned Pitch +
Speaking Rate features, the Relative-minus-Absolute performance difference decreases significantly
as between-speaker target strength increases in both MSP-Podcast and IEMOCAP, under both linear and
quadratic Ridge models. At the pure within-speaker endpoint, Relative outperforms Absolute for
Arousal and Dominance in every confirmatory corpus-by-model cell.

We further test the proposed mechanism rather than only its correlation. In a speaker-center
permutation intervention, the marginal distribution of speaker centers is preserved while center
identity is reassigned across speakers. On raw MSP Arousal and Dominance, using the correct center
substantially outperforms the permuted center; after the target is centered within speaker, that
difference nearly vanishes. Finally, we evaluate practical reference estimation with label-free
speaker enrollment and find that approximately 20 enrollment utterances recover most of the usable
speaker-reference benefit on a fixed MSP downstream pool.

These results support a conditional view of speaker normalization. Stable speaker information can
act as nuisance or useful information depending on the structure of the target. This framing helps
reconcile two active directions in SER: methods that suppress speaker information to improve
speaker-independent robustness, and personalization methods that explicitly exploit speaker
characteristics. Rather than treating one direction as universally correct, our experiments show
that their usefulness depends on which reference frame the task is asking the model to predict.

## Contributions

1. We formulate prosodic normalization as a reference-frame choice by decomposing both acoustic
   features and affect targets into between-speaker and within-speaker components.
2. We introduce a controlled target-reference intervention that continuously varies
   between-speaker target strength without changing acoustic inputs or model architecture.
3. We provide confirmatory cross-corpus evidence on MSP-Podcast and IEMOCAP, using semantically
   matched Pitch + Speaking Rate features and both linear and quadratic regression models.
4. We provide a speaker-center identity intervention showing that stable speaker baselines help
   only when they are correctly aligned to the speaker and the target contains speaker-level
   structure.
5. We evaluate a label-free K-shot deployment setting and quantify how much unlabeled speaker
   reference audio is needed to recover practical baseline utility.
6. We characterize important boundaries: Valence is weak for this low-dimensional prosody
   mechanism; IEMOCAP loudness provenance is not robust enough for the main claim; and strict
   session-disjoint IEMOCAP effects are heterogeneous across only five sessions.

# 2. Related Work

## 2.1 Speaker normalization as nuisance suppression

Speaker normalization has a long history in speech emotion recognition. Bone, Lee, and Narayanan
(2012) proposed an unsupervised cross-corpus arousal-rating framework based on interpretable
prosodic features. Their method explicitly scores pitch, intensity, and voice-quality features
relative to a speaker's neutral-state model, and they note that only a small amount of speaker
reference speech may be sufficient. This is an especially close precursor to any claim involving
speaker-relative prosody, arousal, or enrollment-based reference estimation. Our contribution is
therefore **not** the use of a speaker baseline for affect prediction.

Busso et al. (2013) introduced Iterative Feature Normalization (IFN), estimating normalization
parameters from automatically identified neutral speech and applying the transformation to both
neutral and emotional speech. The objective is to reduce inter-speaker acoustic differences while
preserving inter-emotional variability. Mariooryad and Busso (2014) similarly factorized speaker,
lexical, and expressive variability and showed that compensating speaker and lexical variability
can improve emotion recognition.

Recent neural work preserves the same general motivation while changing the representation.
Gat et al. (2022) used adversarial learning to suppress speaker characteristics from
self-supervised SER representations. Lu et al. (ICASSP 2024) treated speakers as multiple domains
and learned speaker-invariant emotion features through dynamic joint distribution adaptation.
Related speaker-invariant approaches continue to frame speaker-specific acoustic patterns as
domain shift or shortcut information that harms unseen-speaker generalization.

These approaches motivate the question studied here but do not answer it: **when is speaker
variation actually nuisance?** We test whether that answer depends on how much between-speaker
structure is present in the target itself.

## 2.2 Speaker dependence and personalization

A complementary line of work treats individual speaker characteristics as useful information.
Sridhar, Parthasarathy, and Busso (2018) reported stronger speaker-dependent traits in valence
prediction and argued that stronger regularization was needed to learn patterns that generalize
across speakers. Sridhar and Busso (2022) then studied unsupervised personalization of valence
models by retrieving acoustically similar speakers.

Tran, Yin, and Soleymani (2023) proposed personalized continuous emotion recognition using
pretrained speech encoders with speaker embeddings and unsupervised label-distribution
calibration. Triantafyllopoulos and Schuller (2024) used a small set of enrollment utterances to
personalize SER models and studied individual-level fairness. More recent work also explores
few-shot or inference-time speaker adaptation, while speaker-aware multi-task approaches model
speaker- and emotion-specific representations jointly.

This literature establishes that speaker information can be exploited rather than removed, but it
mainly asks whether personalization improves predictive performance. Our analysis asks a different
mechanistic question: **what property of the target determines whether speaker information should
be removed or retained?**

## 2.3 Speaker representations can contain affect information

The assumption that speaker and emotion information can be cleanly separated is also increasingly
questioned. Ulgen et al. (ICASSP 2024) showed that modern speaker embeddings exhibit
emotion-related intra-speaker clusters and used this structure for contrastive SER pretraining.
This provides an important counterpoint to purely speaker-invariant formulations: representations
optimized for speaker information can still contain affective state structure.

Our results are consistent with this broader view. We do not assume that speaker and affect
information occupy disjoint subspaces. Instead, we distinguish stable speaker-level information
from within-speaker deviations and test how each aligns with the target reference frame.

## 2.4 Between-speaker and within-speaker prosodic variation

The distinction between stable between-speaker baselines and within-speaker variation is well
established in phonetics and sociophonetics. Work on speech tempo, for example, has modeled
between-speaker factors and within-speaker phrase-level variation separately and found substantial
variation at both levels (e.g., Quené, 2008; Jacewicz, Fox, and Wei, 2010). Similar considerations
apply to pitch and voice-quality measures.

We adopt this decomposition for an affective prediction question. The experimental element that is
most distinctive here is to apply the same decomposition to the **target**, and then manipulate the
target's between-speaker component while keeping the acoustic observations fixed.

## 2.5 Positioning of the present work

The closest prior work already establishes all of the following ingredients separately:

- speaker-relative prosody can improve arousal estimation;
- speaker normalization can improve cross-speaker emotion recognition;
- speaker-dependent target traits exist;
- personalization can improve continuous emotion prediction;
- speaker embeddings themselves can retain emotion information;
- prosody contains both between- and within-speaker variation.

Our contribution is the link between these observations. We formulate and test a
**target-reference matching principle**: the usefulness of speaker normalization depends on
whether the target asks for within-speaker deviation or retains stable between-speaker structure.

This distinction is tested directly through target intervention and speaker-center permutation,
rather than inferred only from comparing model accuracies before and after normalization.

# 3. Method

## 3.1 Prosodic reference frames

For speaker s and utterance u, let an acoustic prosodic feature be

x_(s,u) = mu_s + delta_(s,u),

where mu_s is a stable speaker reference and delta_(s,u) is the utterance's deviation from that
reference.

We compare three representation families:

- **Absolute:** x_(s,u)
- **Relative:** x_(s,u) - mu_s
- **Hybrid:** [x_(s,u) - mu_s, mu_s]

For the main confirmatory VAD experiment, the acoustic dimensions are Pitch and Speaking Rate,
which have clear semantics in both MSP-Podcast and IEMOCAP. Pitch is represented on a semitone/log
scale, and rate is log transformed. The speaker reference mu_s is the per-speaker marginal median
of each acoustic dimension in the diagnostic oracle analyses.

Loudness is excluded from the cross-corpus confirmatory experiment because the available IEMOCAP
field relative_db has undocumented semantics and does not agree with a physically interpretable
RMS-dB reconstruction in our sensitivity analysis.

## 3.2 Target reference frames

For each VAD target y_(s,u), define the speaker mean ybar_s and global mean ybar. The observed target
can be written as a sum of within-speaker deviation and between-speaker offset.

To manipulate target reference frame without altering acoustic inputs, we define

y_lambda(s,u) =
    ybar
    + [y_(s,u) - ybar_s]
    + lambda [ybar_s - ybar],

where lambda is in {0, 0.25, 0.5, 0.75, 1}.

At lambda = 0, all between-speaker target offsets are removed and the target contains only
within-speaker variation around the global mean. At lambda = 1, the construction recovers the
original raw target exactly. Intermediate lambda values continuously reintroduce the
between-speaker component.

This intervention keeps utterances, acoustic features, speakers, train/test partitions, and model
family fixed. Only the reference frame of the target changes.

## 3.3 Primary hypothesis

Let

Delta(lambda) = CCC_Relative(lambda) - CCC_Absolute(lambda).

The target-reference matching hypothesis predicts

d Delta(lambda) / d lambda < 0

for affect dimensions whose acoustic deviations track within-speaker state while speaker acoustic
centers carry information about between-speaker target structure.

A complementary diagnostic is Hybrid-minus-Relative utility. Stable speaker centers should become
more useful as the target contains more between-speaker information, although nonlinear models may
recover part of that structure implicitly from Absolute or Relative features.

## 3.4 Datasets

### MSP-Podcast

MSP-Podcast provides large-scale natural speech with continuous Valence, Arousal, and Dominance
ratings. Our main MSP analyses use speaker-disjoint folds and speaker-balanced evaluation. The
dataset contains enough speakers to estimate between-speaker target structure, perform permutation
interventions, and evaluate label-free K-shot enrollment.

### IEMOCAP

IEMOCAP provides continuous dimensional ratings together with acted and improvised dyadic speech.
The main cross-corpus confirmatory experiment uses Pitch and Speaking Rate only. IEMOCAP has ten
speakers, which provides an independent dataset but limited power for speaker-level inferential
tests. A stricter leave-one-session-out sensitivity analysis shows substantial session
heterogeneity and is treated as a limitation rather than confirmatory evidence.

### ESD, MEAD, and RAVDESS

These datasets are used for the categorical state-versus-trait motivation. We compare reference
frames for five-class emotion classification and gender classification using low-dimensional
prosody.

## 3.5 Models and evaluation

The confirmatory target-intervention analysis uses two fixed model families:

1. Linear Ridge regression.
2. Quadratic Ridge regression, implemented by standardized degree-2 polynomial expansion followed
   by re-standardization and Ridge regression.

No corpus-specific hyperparameter tuning is used for the target-reference hypothesis.

Continuous affect is evaluated with speaker-balanced Concordance Correlation Coefficient (CCC).
Categorical tasks use Macro-F1. Speaker-disjoint splits are used unless otherwise stated.

## 3.6 Speaker-center permutation intervention

To test whether Hybrid gains arise from the correct speaker reference rather than merely additional
center-valued dimensions, we keep each utterance's Relative features unchanged while reassigning
speaker centers across speakers using a derangement.

The intervention preserves the marginal distribution of center values but destroys
speaker-center identity. Separate derangements are applied within training and test speakers.

If the correct center is useful because it is aligned with speaker-level target structure, then
true-center Hybrid should outperform permuted-center Hybrid for raw targets. The effect should
collapse after the target is centered within speaker.

## 3.7 Label-free K-shot reference estimation

Oracle speaker references assume access to many utterances. For deployment analysis, we estimate
the acoustic center from K unlabeled enrollment utterances from each speaker. Enrollment utterances
are excluded from downstream train/test rows.

A corrected fixed-pool experiment reserves the same 50 enrollment candidates for every K and keeps
all downstream rows identical across K. We compare K in {1, 2, 5, 10, 20, 50} with a full
marginal-speaker oracle.

## 3.8 Representation analysis with WavLM

To test whether a modern self-supervised speech encoder already retains information corresponding
to multiple prosodic reference frames, we probe frozen WavLM-large hidden layers for Absolute
pitch, Relative pitch, and implied speaker baseline pitch.

This analysis addresses decodability, not invariance. High decodability of multiple targets means
that the encoder preserves information related to multiple frames; it does not imply that explicit
Relative features will necessarily improve an emotion classifier on top of WavLM.

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

# 5. Discussion

## 5.1 Normalization is an information transformation

Speaker normalization is often discussed as if it removes unwanted variation while leaving the
task-relevant signal intact. Our results support a different interpretation.

Subtracting a speaker baseline is an information transformation. It suppresses stable
between-speaker information and emphasizes within-speaker deviation. Whether this helps depends on
what the target asks the model to predict.

The target intervention makes this dependence visible. With the acoustic observations held fixed,
changing only the amount of between-speaker structure in the target systematically changes the
Relative-versus-Absolute performance difference. This is the central empirical support for the
reference-frame matching account.

## 5.2 Reconciling speaker invariance and personalization

Two apparently conflicting directions coexist in SER.

Speaker-invariant approaches attempt to suppress speaker information because inter-speaker
variation can become a shortcut and hurt unseen-speaker generalization. Personalization approaches,
by contrast, explicitly condition on speaker characteristics or speaker enrollment.

Our results suggest that neither objective is universally preferable.

For a within-speaker state target, stable speaker information is nuisance by construction.
Relative prosody is therefore well matched to the target.

For a population-level target that contains stable between-speaker structure, the speaker baseline
can become predictive information. Removing it can reduce performance, while exposing it
separately in a Hybrid representation can help.

Thus speaker invariance and personalization can be understood as different choices along the same
reference-frame axis.

## 5.3 Relation to prior speaker-normalized arousal work

The strongest historical precursor is Bone, Lee, and Narayanan (2012), who already demonstrated
that prosodic features scored relative to a speaker's neutral baseline can support robust
cross-corpus arousal rating, including with limited neutral reference data. Busso et al. (2013) and
Mariooryad and Busso (2014) further established speaker normalization and compensation as useful
SER strategies.

The present work should therefore not be positioned as introducing relative prosody or
speaker-baseline enrollment.

The distinction is explanatory rather than procedural. Earlier work primarily asks how to obtain
better emotion recognition by compensating speaker variability. We ask when that compensation
should help or hurt. The controlled lambda intervention manipulates the target reference frame
directly and shows that representation preference moves with it.

## 5.4 Why Arousal and Dominance are clearer than Valence

Low-dimensional prosody strongly supports the proposed mechanism for Arousal and Dominance, but not
for Valence.

On MSP, Valence has lower speaker-level acoustic predictability than Arousal/Dominance in our
simple prosodic feature space, and both raw and within-speaker Valence CCC remain low. This agrees
with the broader SER literature that finds Valence harder to infer from acoustics and more
dependent on speaker, lexical, contextual, or semantic information.

Valence should therefore be treated as a boundary condition rather than forced into the same
low-dimensional prosody explanation.

## 5.5 Why the IEMOCAP effect is smaller than MSP

MSP and IEMOCAP differ substantially in target structure. MSP Arousal and Dominance show large
between-speaker variance fractions, whereas IEMOCAP's are much smaller. Consequently, the
Relative-minus-Absolute slope is much steeper on MSP.

This difference is predicted by the reference-frame account: there is simply less speaker-level
target structure in IEMOCAP for Absolute/Hybrid representations to exploit.

The stricter session-disjoint sensitivity also shows that the IEMOCAP effect is heterogeneous over
only five dyadic sessions. The IEMOCAP evidence should therefore be described as an external
replication of direction under the main speaker-disjoint protocol, not as proof of universal
session-level consistency.

## 5.6 Feature semantics matter

A key sensitivity result concerns loudness.

The available IEMOCAP relative_db field behaves differently from an RMS-dB reconstruction, and the
loudness-based target-reference result does not survive this substitution. This demonstrates a
general methodological lesson: a reference-frame study is only meaningful if the physical meaning
of the feature and its baseline are clearly defined.

For this reason, the main cross-corpus result uses only Pitch and Speaking Rate, whose definitions
are transparent in both corpora.

## 5.7 Deployment implications

The oracle decomposition is useful for mechanism analysis but cannot be assumed at inference time.

The K-shot experiments show that useful speaker references can be estimated from unlabeled
enrollment audio. On the corrected MSP fixed-pool analysis, approximately 20 utterances recover
most of the practical Hybrid benefit.

This result suggests that reference-frame-aware systems need not require emotion labels or a
manually collected neutral state. However, the amount and composition of enrollment data remain
corpus-dependent. IEMOCAP's non-monotonic K-shot mechanism recovery illustrates that more accurate
physical center estimation does not guarantee monotonic downstream gains.

## 5.8 Implications for representation learning

The WavLM probes show that Absolute pitch, Relative pitch, and implied speaker baseline are all
strongly decodable from hidden representations, with different layer profiles.

This suggests that modern SSL encoders do not simply become speaker-invariant in a binary sense.
They can retain multiple overlapping forms of state and trait information.

A useful next step for representation learning is therefore not merely to ask whether speaker
information is present, but whether downstream readouts emphasize the reference frame appropriate
for the target.

## 5.9 Limitations

The present study has several important limitations.

1. Continuous-VAD external validation is limited to MSP-Podcast and IEMOCAP.
2. IEMOCAP contains only ten speakers and five dyadic sessions; strict session-level replication is
   heterogeneous.
3. The main interpretable acoustic space is intentionally low-dimensional. More complex prosodic
   contours, voice quality, lexical content, and multimodal cues may interact with reference frame
   differently.
4. The marginal speaker center is a diagnostic oracle. K-shot experiments reduce this deployment
   gap but do not solve reference estimation for all settings.
5. The target intervention is a controlled statistical construction. It establishes a mechanism
   within the observed dataset but does not imply that naturally occurring applications will have
   a known lambda.
6. Gender is used as a strong stable-trait probe, but speaker-ID results show that Relative prosody
   only modestly suppresses full speaker identity.
7. The study does not establish that every affect dimension or every corpus follows the same effect
   magnitude.

## 5.10 Broader methodological implication

SER evaluation often treats a label such as Arousal as if its meaning were independent of
population structure. Our results suggest that evaluation should state whether the intended target
is:

- population-level absolute affect,
- deviation from a person's own baseline,
- or a combination of both.

These targets are not interchangeable, and they can prefer different acoustic representations.

The main conclusion is therefore not a recommendation to normalize or not normalize. It is a
recommendation to make the reference frame explicit.

# 6. Conclusion

We studied speaker normalization as a reference-frame choice rather than a universally beneficial
preprocessing operation.

Across categorical state-versus-trait tasks, continuous VAD decomposition, a controlled
between-speaker target intervention, speaker-center permutation, and label-free enrollment, the
results support one consistent principle: the usefulness of stable speaker information depends on
the reference frame of the target.

When the target represents within-speaker affect deviation, speaker-relative prosody is consistently
advantageous for Arousal and Dominance. As stable between-speaker target structure is introduced,
Relative becomes less favorable and speaker baseline information can become useful. This coupling
replicates across MSP-Podcast and IEMOCAP using semantically matched Pitch + Speaking Rate
features and both linear and quadratic models.

The practical implication is not that speaker normalization should always be applied, nor that it
should always be avoided. Instead, SER systems should make an explicit choice about whether the
prediction target is population-relative or person-relative, and construct their acoustic
representation accordingly.

In short:

**Representation reference frame should match target reference frame.**

---

## Draft Reference Notes

These notes are for drafting and must be converted to the final bibliography style before
submission.

1. Bone, D., Lee, C.-C., Narayanan, S. S. (2012).
   "A Robust Unsupervised Arousal Rating Framework using Prosody with Cross-Corpora Evaluation."
   Interspeech 2012. DOI: 10.21437/Interspeech.2012-123.

2. Busso, C., Mariooryad, S., Metallinou, A., Narayanan, S. S. (2013).
   "Iterative Feature Normalization Scheme for Automatic Emotion Detection from Speech."
   IEEE Transactions on Affective Computing, 4(4), 386-397.
   DOI: 10.1109/T-AFFC.2013.26.

3. Mariooryad, S., Busso, C. (2014).
   "Compensating for speaker or lexical variabilities in speech for emotion recognition."
   Speech Communication, 57, 1-12.
   DOI: 10.1016/j.specom.2013.07.011.

4. Sridhar, K., Parthasarathy, S., Busso, C. (2018).
   "Role of Regularization in the Prediction of Valence from Speech."
   Interspeech 2018.

5. Gat, I., Aronowitz, H., Zhu, W., Morais, E., Hoory, R. (2022).
   "Speaker Normalization for Self-Supervised Speech Emotion Recognition."
   ICASSP 2022, 7342-7346.
   DOI: 10.1109/ICASSP43922.2022.9747460.

6. Sridhar, K., Busso, C. (2022).
   "Unsupervised Personalization of an Emotion Recognition System: The Unique Properties of the
   Externalization of Valence in Speech."
   IEEE Transactions on Affective Computing.
   DOI: 10.1109/TAFFC.2022.3187336.

7. Tran, M., Yin, Y., Soleymani, M. (2023).
   "Personalized Adaptation with Pre-trained Speech Encoders for Continuous Emotion Recognition."
   Interspeech 2023, 636-640.
   DOI: 10.21437/Interspeech.2023-2170.

8. Triantafyllopoulos, A., Schuller, B. (2024).
   "Enrolment-based personalisation for improving individual-level fairness in speech emotion
   recognition."
   Interspeech 2024, 3729-3733.
   DOI: 10.21437/Interspeech.2024-98.

9. Lu, C., Zong, Y., Lian, H., Zhao, Y., Schuller, B., Zheng, W. (2024).
   "Improving Speaker-independent Speech Emotion Recognition Using Dynamic Joint Distribution
   Adaptation."
   ICASSP 2024.
   DOI: 10.1109/ICASSP48485.2024.10447452.

10. Ulgen, I. R., Du, Z., Busso, C., Sisman, B. (2024).
    "Revealing Emotional Clusters in Speaker Embeddings: A Contrastive Learning Strategy for
    Speech Emotion Recognition."
    ICASSP 2024.
    DOI: 10.1109/ICASSP48485.2024.10447060.

11. Jacewicz, E., Fox, R. A., Wei, L. (2010).
    "Between-speaker and within-speaker variation in speech tempo of American English."
    Journal of the Acoustical Society of America, 128(2), 839-850.
    DOI: 10.1121/1.3459842.

12. Quené, H. (2008).
    "Multilevel modeling of between-speaker and within-speaker variation in spontaneous speech
    tempo."
    Journal of the Acoustical Society of America, 123(2), 1104-1113.
    DOI: 10.1121/1.2821762.

13. Shi, X., Li, X., Toda, T. (2025).
    "Speaker-Aware Multi-Task Learning for Speech Emotion Recognition."
    Interspeech 2025.
    DOI: 10.21437/Interspeech.2025-1439.
