# 2. Related Work

## 2.1 Speaker normalization as nuisance suppression

Speaker normalization has a long history in speech emotion recognition. Bone, Lee, and Narayanan
(2012), and the extended IEEE T-AFFC study by Bone et al. (2014), are especially close precursors.
Their cross-corpus arousal framework constructs a speaker-specific baseline model for a small set of
interpretable vocal features and scores utterances relative to that baseline. The journal study
explicitly presents a scale-continuous arousal measure, prefers neutral speech for baseline
estimation, and provides evidence that labeled neutral data are not always necessary. These works
already establish speaker-relative prosody, continuous arousal scoring, cross-corpus evaluation,
and limited-reference baseline estimation. Our contribution is therefore **not** the use of a
speaker baseline, relative prosody, continuous arousal prediction, or small enrollment sets.

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

We adopt this decomposition for an affective prediction question. Importantly, person-mean
centering and the statistical separation of within-person and between-person variation are
themselves standard tools in multilevel and affect-dynamics research. We therefore do not claim
novelty for target centering as a statistical operation.

The distinctive experimental step is to connect the two reference-frame choices: we continuously
manipulate the target's between-speaker component while holding acoustic observations, splits, and
model family fixed, and then test whether the preferred **acoustic representation** changes with
that target intervention.

## 2.5 Positioning of the present work

The closest prior work already establishes all of the following ingredients separately:

- speaker-relative prosody can improve arousal estimation;
- speaker normalization can improve cross-speaker emotion recognition;
- speaker-dependent target traits exist;
- personalization can improve continuous emotion prediction;
- speaker embeddings themselves can retain emotion information;
- prosody contains both between- and within-speaker variation.

Our contribution is the controlled link between these observations. Neither side of the link is
new in isolation: speaker-relative normalization is established in SER, and within-person versus
between-person centering is established in multilevel affect research. We formulate and test a
**target-reference matching principle**: the usefulness of speaker normalization depends on
whether the target asks for within-speaker deviation or retains stable between-speaker structure.

The empirical contribution is that target reference frame is treated as an intervention variable.
We vary the between-speaker target component continuously while keeping the acoustic observations
and models fixed, test the resulting Relative-versus-Absolute slope, and then use speaker-center
permutation to isolate whether correct speaker-center identity explains raw-target utility.
