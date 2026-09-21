# Draft Introduction + Related Work

## Introduction

Speech emotion recognition systems must separate affective variation from the many other sources
of variability carried by the voice. One of the oldest and most persistent of these sources is
speaker identity. Pitch range, loudness, speaking rate, and other prosodic statistics vary
substantially across speakers even in the absence of a change in affect. As a result, speaker
normalization has long been treated as a natural preprocessing step for emotion recognition.

This view is intuitive but incomplete. Stable speaker variation is not always nuisance. A target
label can itself contain stable between-speaker structure. For example, two speakers may receive
systematically different arousal or dominance ratings even when both show similar
within-speaker deviations from their own baselines. In such a setting, aggressively removing the
speaker baseline may also remove information that is predictive of the target. Conversely, when
the task is explicitly to track a speaker's deviation from their own typical state, the same
baseline becomes a nuisance.

This suggests that prosodic normalization should be viewed as a **reference-frame choice** rather
than a universally beneficial invariance operation. Let an utterance-level prosodic measurement be

x_su = μ_s + δ_su,

where μ_s is a stable speaker component and δ_su is a within-speaker deviation. Likewise, let a
continuous affect target be decomposed as

y_su = ȳ_s + ε_su,

where ȳ_s captures stable between-speaker target structure and ε_su captures within-speaker
variation. Absolute prosody retains both μ_s and δ_su; speaker-relative prosody emphasizes δ_su;
and a Hybrid representation exposes the two components separately. Under this view, the preferred
representation should depend on how much of the target lies in ȳ_s versus ε_su.

We test this **target-reference matching** hypothesis across categorical and continuous affect
tasks. First, using explicit prosodic features on ESD, MEAD, and RAVDESS, we show that
speaker-relative pitch improves categorical emotion recognition while sharply reducing gender
decodability, illustrating that normalization redistributes state- and trait-related information
rather than simply removing noise. We then move to continuous affect and explicitly manipulate
the amount of between-speaker structure in Valence-Arousal-Dominance targets while holding
acoustic inputs, speakers, and prediction models fixed. Across MSP-Podcast and IEMOCAP, and using
a semantically aligned Pitch+Speaking-Rate feature set, increasing between-speaker target strength
consistently makes Relative prosody less favorable than Absolute prosody for Arousal and Dominance
under both linear and quadratic models. At the pure within-speaker endpoint, Relative is better in
every confirmatory corpus-target-model cell.

We further test mechanism rather than correlation. When the correct speaker center is replaced
with a center drawn from another speaker while preserving the center distribution, most of the raw
MSP Arousal/Dominance Hybrid advantage disappears; the same intervention has almost no effect once
the target is centered within speaker. Finally, we show that the practical reference need not be
oracle information: a small unlabeled enrollment set recovers most of the usable speaker-center
benefit on MSP.

Together, these findings support a simple principle:

> **Representation reference frame should match target reference frame.**

This principle reconciles two strands of prior work that otherwise appear contradictory: one
treats speaker variability as a nuisance to normalize away, while another explicitly models or
personalizes to speaker information. Our results suggest that neither strategy is universally
correct. Whether speaker information should be suppressed or preserved depends on the structure
of the prediction target.

### Contributions

1. We formulate prosodic normalization as a reference-frame choice and jointly decompose acoustic
   prosody and affect targets into between-speaker and within-speaker components.
2. We introduce a controlled target intervention that varies between-speaker target strength while
   keeping acoustic inputs and model family fixed, providing direct evidence for target-reference
   matching across MSP and IEMOCAP.
3. We show that correct speaker-center identity is causally relevant for raw Arousal/Dominance
   prediction but becomes nearly irrelevant for within-speaker centered targets.
4. We demonstrate that the mechanism is robust to linear versus quadratic access and to a
   semantically matched Pitch+Speaking-Rate feature set.
5. We quantify deployment requirements through label-free K-shot speaker reference estimation and
   connect explicit prosodic reference frames to frozen WavLM representations.

## Related Work

### Speaker normalization in speech emotion recognition

Speaker normalization has a long history in SER. Earlier work normalized pitch and other prosodic
features to reduce inter-speaker variability, often motivated by the observation that raw acoustic
values are difficult to compare across speakers. Busso, Mariooryad, and collaborators formalized
speaker and lexical variability compensation and showed that normalization can improve emotion
classification. Later work extended the same idea to learned representations: Gat et al. used
adversarial speaker normalization with self-supervised speech representations, explicitly treating
speaker characteristics as shortcuts that may hurt generalization.

Our work differs in the underlying question. Rather than assuming that speaker variation is
nuisance, we ask when it is nuisance. We explicitly preserve or remove the speaker baseline and
show that its utility changes with the between-speaker structure of the target.

### Speaker-relative affect modeling

Several prior studies directly exploit speaker-relative information. Bone et al. constructed
speaker-specific baselines for pitch, intensity, and spectral-energy features and produced
continuous arousal ratings relative to those baselines. Their framework showed that little
speaker-specific neutral data can be sufficient for useful arousal estimation, and subsequent
work emphasized that arousal ratings should often be interpreted relatively rather than
absolutely. Cao et al. used speaker-sensitive ranking, treating each speaker as a separate query
and showing that within-speaker ranking improves categorical emotion recognition. More recently,
ASER uses speaker-relative prosodic coding as an interpretable intermediate attribute in a
generative multi-task SER framework.

These studies establish the value of relative information, but they do not test when a relative
frame should be preferred over an absolute or hybrid frame. The central distinction in our work is
that we manipulate the **target reference frame** itself and show that representation preference
changes systematically as a consequence.

### Speaker-aware and personalized emotion recognition

A separate line of work moves in the opposite direction and explicitly models speaker
information. Speaker-aware multi-task models learn speaker-specific and emotion-specific
representations, while personalization studies adapt affect models to target speakers.
Dimensional-affect work has also reported stronger speaker-dependent traits for some affect
dimensions, especially valence, motivating stronger regularization or speaker adaptation.

These results demonstrate that speaker information can be useful, but they leave open a unifying
criterion for deciding when it should be preserved. Our target-reference matching account offers
such a criterion: stable speaker information is useful to the extent that the target itself
contains stable between-speaker structure.

### Between-speaker and within-speaker variability

Speech production research has long separated between-speaker and within-speaker variability in
prosodic measures such as speaking rate and F0. This distinction motivates our acoustic
decomposition but does not by itself specify how those components should be used for affect
prediction. We extend the same decomposition to the target side and directly test the interaction
between representation and target reference frames.
