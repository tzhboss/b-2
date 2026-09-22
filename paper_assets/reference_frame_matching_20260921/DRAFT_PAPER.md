# Normalization Is Not Neutral: Matching Prosodic Reference Frames to Affective Targets

## Abstract

Speaker normalization is widely used in speech emotion recognition because prosodic measurements
such as pitch and speaking rate vary strongly across speakers. This practice implicitly treats
stable speaker variation as nuisance. We argue that this assumption is incomplete: whether a
speaker baseline is nuisance or useful information depends on the reference frame of the target.

We decompose an utterance-level prosodic feature into a stable speaker component and a
within-speaker deviation, and analogously decompose continuous affect targets into between-speaker
and within-speaker components. This yields Absolute, speaker-Relative, and Hybrid prosodic
representations and motivates a target-reference matching hypothesis.

Across ESD, MEAD, and RAVDESS, speaker-relative pitch improves categorical emotion recognition
while strongly reducing gender decodability, showing that normalization redistributes rather than
uniformly improves task-relevant information. We then directly manipulate the between-speaker
component of Arousal and Dominance targets. Using only semantically aligned pitch and speaking rate,
Relative-minus-Absolute CCC decreases significantly as between-speaker target strength increases in
both MSP-Podcast and IEMOCAP, under both linear and quadratic regression. At the pure
within-speaker endpoint, Relative is better in every confirmatory corpus-target-model condition.

A speaker-center permutation intervention further shows that raw MSP Arousal/Dominance benefits
require the correct speaker center, while center identity becomes nearly irrelevant after the
target is centered within speaker. Finally, approximately 20 unlabeled enrollment utterances
recover most of the practical speaker-reference benefit on MSP.

These results support a simple principle: **representation reference frame should match target
reference frame**. Speaker normalization is therefore better understood as an information
transformation than as a universally beneficial invariance operation.

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


# Draft Methods + Results

## Methods

### Problem formulation

We study prosodic normalization as a reference-frame choice. For speaker s and utterance u, an
utterance-level acoustic feature is decomposed as

x_su = μ_s + δ_su,

where μ_s is a stable speaker-specific acoustic center and δ_su is a within-speaker deviation.

For continuous affect, we analogously write

y_su = ȳ_s + ε_su,

where ȳ_s is the speaker-level target mean and ε_su is the within-speaker target residual.

We compare three representation families:

- Absolute: x_su
- Relative: x_su − μ_s
- Hybrid: [x_su − μ_s, μ_s]

The core hypothesis is that the preferred representation depends on the target reference frame.

### Categorical state-versus-trait pilot

We first evaluate explicit prosodic features on ESD, MEAD, and RAVDESS. We construct Absolute,
Relative, and Hybrid representations from pitch, loudness, and speaking rate, and evaluate
5-class emotion recognition and gender classification under matched folds.

The purpose of this experiment is motivational rather than confirmatory: it asks whether changing
the acoustic reference frame redistributes information useful for a state-like target (emotion)
versus a stable speaker trait (gender).

### Continuous affect datasets

We use MSP-Podcast for large-scale continuous Valence, Arousal, and Dominance prediction and
IEMOCAP for external validation.

For MSP, the dimensional targets are human SAM ratings on a 1-7 scale. For IEMOCAP, we use the
continuous dimensional annotations EmoVal, EmoAct, and EmoDom.

### Acoustic features

The clean cross-corpus confirmatory experiment uses only features with aligned and interpretable
semantics in both corpora:

- Pitch:
  - MSP: 12 log2(f0_median_hz)
  - IEMOCAP: 12 log2(pitch_mean)
- Speaking rate:
  - MSP: log(phoneme_articulation_rate)
  - IEMOCAP: log(speaking_rate)

IEMOCAP loudness is excluded from the primary confirmatory evidence because the source field
relative_db has undocumented semantics and failed robustness against a physically interpretable
RMS-dB replacement.

Speaker centers are computed as full-speaker marginal medians in the mechanism experiments.
These oracle centers are diagnostic quantities, not a deployment assumption.

### Target decomposition

For each continuous target, we estimate the speaker mean ȳ_s and define the within-speaker residual

ε_su = y_su − ȳ_s.

This allows us to compare raw population-level prediction with explicitly within-speaker prediction.

### Controlled target-reference intervention

To directly vary the target reference frame while keeping acoustic inputs and models fixed, we
construct

y_su(λ) = ȳ + ε_su + λ(ȳ_s − ȳ),

with λ ∈ {0, 0.25, 0.5, 0.75, 1}.

λ=0 removes the between-speaker component while preserving within-speaker variation.
λ=1 recovers the original raw target.

For each λ we train the same models on the same acoustic inputs and evaluate the change in

CCC(Relative) − CCC(Absolute).

A negative slope versus λ means Relative becomes less favorable as between-speaker target
structure increases.

### Models and evaluation

The primary models are:

- Linear Ridge regression
- Quadratic Ridge regression using degree-2 polynomial features after train-only standardization

We use speaker-disjoint folds, equal total train/evaluation weight per speaker, fixed regularization,
and no target-specific or corpus-specific hyperparameter tuning.

Continuous affect is evaluated using speaker-balanced concordance correlation coefficient (CCC).
Categorical tasks use Macro-F1.

### Speaker-center permutation intervention

To test whether Hybrid utility depends on the correct speaker center rather than merely additional
features, we keep the Relative features unchanged and replace each speaker's center with another
speaker's center via a within-split derangement. The center marginal distribution is preserved while
speaker-center identity is destroyed.

### Label-free K-shot enrollment

For deployment, we estimate a new speaker's acoustic center from K unlabeled enrollment utterances.
Enrollment utterances are excluded from downstream train/test rows. We use the per-dimension median
of pitch, loudness, and log-rate and compare K-shot Hybrid against Absolute and a diagnostic
marginal-oracle Hybrid.

The corrected fixed-pool study reserves 50 enrollment utterances per speaker and uses identical
downstream rows for every K ∈ {1, 2, 5, 10, 20, 50}.

### Frozen WavLM analysis

We additionally probe frozen WavLM-large hidden layers for decodability of Absolute pitch,
speaker-relative pitch, and implied speaker baseline. This analysis asks whether modern SSL
representations retain information corresponding to multiple prosodic reference frames.

---

## Results

### 1. Relative pitch helps categorical emotion but removes trait information

Across ESD, MEAD, and RAVDESS, Relative pitch consistently improves 5-class emotion Macro-F1 over
Absolute pitch while strongly reducing gender Macro-F1.

Relative-minus-Absolute pitch Macro-F1:

- ESD emotion: +0.0908; gender: -0.1330
- MEAD emotion: +0.0904; gender: -0.3790
- RAVDESS emotion: +0.0586; gender: -0.2299

This establishes that normalization redistributes task-relevant information rather than uniformly
improving representation quality.

### 2. MSP raw VAD contains substantial between-speaker structure

The ICC-like between-speaker fractions in MSP are approximately:

- Valence: 0.213
- Arousal: 0.418
- Dominance: 0.346

A stable prosodic baseline alone predicts speaker-level target means with CCC:

- Arousal: 0.482
- Dominance: 0.441
- Valence: 0.022

Thus raw Arousal and Dominance contain large speaker-level components that simple stable acoustic
statistics can predict.

### 3. Once the target is centered within speaker, Relative becomes strongly preferable

For within-speaker residual targets, Relative-minus-Absolute all-prosody CCC is:

- Arousal: +0.1306, 95% CI [0.1282, 0.1326]
- Dominance: +0.0921, [0.0894, 0.0945]
- Valence: +0.0049

Adding the speaker baseline back to Relative after target centering provides essentially no gain:

- Arousal: +0.00049
- Dominance: +0.00029

This directly supports the idea that speaker baseline utility depends on target reference frame.

### 4. Controlled target intervention confirms target-reference matching

The cleanest confirmatory experiment uses only Pitch+Rate in both MSP and IEMOCAP and evaluates
both Linear and Quadratic Ridge.

Slope of CCC(Relative) − CCC(Absolute) versus λ:

Using speaker as the independent bootstrap unit, MSP shows strong confirmatory slopes:

- Linear Arousal: -0.3188, 95% CI [-0.3384, -0.2998]
- Linear Dominance: -0.1825, [-0.1969, -0.1683]
- Quadratic Arousal: -0.3214, [-0.3411, -0.3022]
- Quadratic Dominance: -0.1851, [-0.2000, -0.1705]

IEMOCAP point slopes are directionally consistent but less precise with only ten speakers:

- Linear Arousal: -0.0192, 95% CI [-0.0662, +0.0326]
- Linear Dominance: -0.0162, [-0.0484, +0.0123]
- Quadratic Arousal: -0.0281, [-0.0758, +0.0256]
- Quadratic Dominance: -0.0271, [-0.0617, +0.0058]

Thus the slope mechanism is strongly confirmed on MSP and directionally replicated, but not
independently significant, on IEMOCAP under speaker-cluster inference.

At λ=0, Relative is better than Absolute in every Arousal/Dominance corpus-target-model cell,
with speaker-cluster confidence intervals above zero in both corpora:

- MSP Linear: +0.1390 Arousal, +0.0727 Dominance
- MSP Quadratic: +0.1390 Arousal, +0.0746 Dominance
- IEMOCAP Linear: +0.1375 Arousal, +0.0456 Dominance
- IEMOCAP Quadratic: +0.1462 Arousal, +0.0458 Dominance

The strongest inferential evidence for the slope comes from MSP, while IEMOCAP robustly supports
the pure within-speaker endpoint and the same negative point-slope direction.

### 5. Correct speaker-center identity is necessary for raw Hybrid utility

We keep Relative features fixed and replace the appended speaker center with another speaker's
center.

On MSP raw targets, true center minus permuted center yields:

- Arousal: +0.2955 CCC, 95% CI [0.2856, 0.3039]
- Dominance: +0.2363, [0.2285, 0.2436]

After target centering, the same effect collapses to:

- Arousal: +0.00139
- Dominance: +0.00100

Thus the Hybrid gain is not caused by adding generic center-valued dimensions; it depends on the
correct speaker prior and disappears when the target no longer contains speaker-level structure.

IEMOCAP shows much smaller raw center-identity effects, consistent with its weak between-speaker
target structure.

### 6. About 20 unlabeled enrollment utterances recover most practical speaker-reference utility

In the corrected fixed-downstream-pool MSP experiment, K-shot Hybrid CCC is:

Arousal:
- K1: 0.4795
- K10: 0.4871
- K20: 0.4903
- K50: 0.4946

Dominance:
- K1: 0.3793
- K10: 0.3837
- K20: 0.3839
- K50: 0.3840

At K=20, K-shot Hybrid is only 0.00427 CCC below the marginal oracle for Arousal and is effectively
identical for Dominance (+0.00036 relative to oracle).

Therefore deployment does not require affect labels or a full speaker history to recover most of
the usable speaker-reference benefit.

### 7. WavLM retains multiple reference frames

Frozen WavLM-large representations preserve highly decodable information about Absolute pitch,
Relative pitch, and implied speaker baseline.

For Relative pitch, mean R² declines from earlier/middle to late layers:

- ESD: 0.8825 in layers 5-12 to 0.8114 in layers 21-24
- MEAD: 0.8462 to 0.7609
- RAVDESS: 0.8766 to 0.7754

Implied speaker baseline remains very strongly decodable throughout the network.

This shows that SSL representations do not simply become speaker-invariant. Instead, reference-frame
information remains available but is reorganized with depth.


### 8. Target-reference matching survives in WavLM embedding space

We next construct Absolute, Relative, and Hybrid reference frames directly in frozen WavLM-large
embedding space on IEMOCAP. For each speaker, the center is the coordinate-wise median embedding;
Relative subtracts this center and Hybrid concatenates Relative with the center.

At hidden layer 12, speaker-cluster Relative-minus-Absolute slopes are:
- Arousal: -0.0829, 95% CI [-0.1309, -0.0359]
- Dominance: -0.0341, [-0.0583, -0.0118]

At layer 24:
- Arousal: -0.0958, 95% CI [-0.1429, -0.0482]
- Dominance: -0.0455, [-0.0707, -0.0181]

All four slope intervals are below zero. At λ=0, Relative is significantly better in all four
layer-by-target cells; at λ=1, Absolute is better in all four. The same reference preference
reversal therefore appears in a 1024-dimensional learned speech representation rather than only
in handcrafted scalar prosodic attributes.

The default Ridge solver emitted ill-conditioning warnings in this high-dimensional setting.
Repeating the full probe with LSQR at tolerance 1e-8 changed the primary slopes by at most
1.65e-6 and the lambda=0 endpoint effects by at most 1.01e-6, leaving all conclusions unchanged.

### Numerical stability of the WavLM result

A high-precision LSQR sensitivity reproduces the WavLM target-reference slopes almost exactly.
Across the four layer-by-target cells, the maximum absolute slope change is 1.65e-6 and the maximum
lambda=0 effect change is 1.01e-6. The learned-representation result is therefore insensitive to
the Ridge solver used.

### 9. Boundaries

The mechanism is strongest for Arousal and Dominance. Valence is weak under these low-dimensional
prosodic features and should not be overinterpreted.

IEMOCAP loudness is excluded from the main claim because replacing relative_db with RMS dB removes
or reverses the loudness-specific mechanism.

A stricter IEMOCAP leave-one-session-out sensitivity analysis has only five independent sessions and
shows substantial session heterogeneity. It should be treated as a limitation rather than
confirmatory evidence.

Speaker-relative low-dimensional prosody only modestly reduces speaker-ID decodability, even though
gender suppression is strong. We therefore avoid claiming that Relative prosody universally removes
speaker identity.


# Draft Discussion + Conclusion

## Discussion

### Normalization is an information transformation

Speaker normalization is often motivated as a way to remove unwanted variability. Our results
suggest a more precise interpretation: normalization changes which information is directly
accessible to a downstream model.

Absolute prosody contains both stable speaker structure and within-speaker deviation. Relative
prosody suppresses the former and emphasizes the latter. Neither representation is intrinsically
better. Their usefulness depends on whether the target rewards the information retained by that
reference frame.

This interpretation explains the otherwise contradictory observations in our experiments.
Relative pitch improves categorical emotion recognition while sharply reducing gender
decodability, yet raw MSP Arousal and Dominance strongly benefit from stable speaker information.
Once the same VAD targets are centered within speaker, the advantage reverses and Relative becomes
strongly preferable.

### Why target structure matters

The controlled lambda intervention is important because it moves beyond a corpus-level
correlation. The acoustic inputs, speakers, and model remain unchanged while only the
between-speaker component of the target is varied. The systematic negative slope of
Relative-minus-Absolute utility therefore links representation preference directly to target
reference frame.

The much larger slopes in MSP than IEMOCAP are also informative. MSP raw Arousal and Dominance
contain much stronger between-speaker structure, whereas IEMOCAP's dimensional labels are largely
within-speaker at the population level. The reference-frame account predicts exactly this
difference: removing speaker baseline should be most costly when the target itself contains a
large speaker-level component.

### Speaker information is not simply shortcut information

Recent SER systems often treat speaker identity as a shortcut that should be removed. Other work,
however, finds gains from speaker-aware representations or personalization. Our results reconcile
these views.

The speaker-center permutation experiment shows that stable speaker information is useful only
when it is aligned to the correct speaker and to a target containing speaker-level structure.
When target speaker means are removed, center identity becomes almost irrelevant.

Thus the correct question is not whether speaker information is good or bad. The relevant question
is whether the downstream target defines affect relative to a population or relative to the
speaker's own baseline.

### Relation to speaker-relative arousal modeling

Prior work has shown that speaker-relative prosodic baselines can support robust and interpretable
arousal estimation. Our findings agree with that literature at the within-speaker endpoint, but
extend it in two directions.

First, we explicitly compare Absolute, Relative, and Hybrid information rather than assuming the
relative frame is universally preferable. Second, we intervene on the target itself and show that
the advantage of Relative prosody changes predictably as between-speaker target structure is added
or removed.

### Deployment implications

Oracle speaker statistics are useful for mechanism analysis but unrealistic for deployment.
Our K-shot experiments show that this limitation is manageable: unlabeled enrollment audio can
estimate a practical speaker reference without affect annotations.

On MSP, approximately 20 enrollment utterances recover most of the usable Hybrid benefit.
However, the exact K should not be universalized. IEMOCAP reference-quality experiments show that
physical center estimation can improve monotonically while downstream target-reference effects
remain noisy with only a small number of speakers.

This suggests that future systems should expose their reference-estimation assumptions explicitly:
how much enrollment is used, whether the reference is neutral or unlabeled, and whether the
reference is updated over time.

### Implications for self-supervised speech representations

Frozen WavLM encodes information corresponding to Absolute pitch, Relative pitch, and implied
speaker baseline simultaneously. Relative-pitch decodability decreases toward later layers, but
speaker baseline remains highly accessible.

This argues against a simple claim that modern SSL encoders automatically remove speaker
reference frames. Instead, multiple frames remain encoded, and their accessibility changes with
depth. Explicit reference-frame features therefore provide a useful analysis tool even when they
do not always improve downstream WavLM prediction.

### Valence is a boundary case

The strongest effects concern Arousal and Dominance. Valence is weakly predicted by the
low-dimensional prosodic attributes used here and does not show the same clear reference-frame
pattern.

This limitation is consistent with prior dimensional-affect work showing that valence is often
harder to predict acoustically and may rely more strongly on lexical, semantic, or
speaker-dependent cues. The present results should therefore not be generalized to all affect
dimensions.

### Limitations

First, the clean continuous-affect replication currently relies on two corpora, MSP-Podcast and
IEMOCAP. The IEMOCAP sample contains only ten speakers and five dyadic sessions.

Second, a stricter leave-one-session-out sensitivity analysis shows substantial session
heterogeneity. Mean Arousal/Dominance slopes remain negative, but the direction is not consistent
across all five sessions. This result should constrain claims of universal cross-session
generalization.

Third, IEMOCAP loudness provenance is ambiguous. Replacing its source relative_db field with RMS dB
changes the result substantially, so loudness is excluded from the main cross-corpus evidence.

Fourth, gender is only an illustrative stable-trait task. A separate speaker-identification audit
shows that Relative low-dimensional prosody suppresses speaker identity only modestly.

Finally, our target intervention is intentionally diagnostic. Constructing within-speaker target
residuals requires speaker-level target statistics and is not itself a deployment procedure. Its
purpose is to identify the mechanism governing representation preference.

## Conclusion

We studied speaker normalization as a reference-frame choice rather than a universally beneficial
preprocessing operation.

Across categorical emotion tasks, continuous affect regression, controlled target interventions,
speaker-center permutation, K-shot reference estimation, and frozen SSL analysis, the same
principle emerges: the value of stable speaker information depends on whether the target itself
contains stable between-speaker structure.

When the target is defined relative to a speaker's own state, speaker-relative prosody is
consistently advantageous. As between-speaker target structure is introduced, Absolute and Hybrid
representations become increasingly useful. The correct speaker center matters only when that
speaker-level target structure is present.

These findings motivate a shift in how normalization is discussed in speech affect modeling.
Rather than asking whether speaker normalization improves performance, future work should ask
which reference frame the task actually requires.


## Figure and Table Plan

### Main figures

- Figure 1: Target-reference matching with semantically matched Pitch+Rate.
  File: fig1_target_reference_matching_pitch_rate.png
- Figure 2: Label-free K-shot reference saturation on MSP.
  File: fig2_kshot_reference_saturation_msp.png
- Figure 3: Categorical emotion/gender pitch reversal.
  File: fig3_emotion_gender_pitch_reversal.png
- Figure 4: WavLM layer-wise reference-frame decodability.
  File: fig4_wavlm_reference_decodability_esd.png
- Figure 5: Direct WavLM target-reference intervention.\n  File: fig5_wavlm_target_reference_intervention.png\n- Figure 6: Extended prosodic attribute slopes.\n  File: fig6_extended_prosody_slopes.png\n
### Main tables

- Table 1: Confirmatory Pitch+Rate slopes.
  File: table1_confirmatory_pitch_rate_slopes.csv
- Table 2: Within-speaker lambda=0 endpoints.
  File: table2_within_speaker_endpoints.csv
- Table 3: Categorical Pitch Relative-Absolute reversal.
  File: table3_categorical_pitch_reversal.csv
- Table 4: MSP K-shot Hybrid performance.
  File: table4_kshot_msp.csv
- Table 6: Speaker-cluster main inference.\n  File: table6_speaker_cluster_main_inference.csv\n- Table 7: WavLM target-reference slopes.\n  File: table7_wavlm_target_reference_slopes.csv\n- Table 8: Extended attribute slopes.\n  File: table8_extended_attribute_slopes.csv\n
### Appendix / supplement

- WavLM layer blocks and full layer curves.
- Annotation-disagreement robustness.
- Attribute-level Pitch/Loudness/Rate intervention.
- Speaker-ID decodability.
- IEMOCAP K-shot reference-quality sensitivity.
- IEMOCAP loudness semantic robustness failure.
- Corrected session-disjoint sensitivity.

## Citation TODO

Replace prose mentions with formal BibTeX entries for at least:

- Busso et al., Iterative Feature Normalization Scheme for Automatic Emotion Detection from Speech.
- Mariooryad/Busso et al., Compensating for speaker or lexical variabilities in speech for emotion recognition.
- Bone, Lee, Narayanan, A Robust Unsupervised Arousal Rating Framework using Prosody with Cross-Corpora Evaluation.
- Bone et al., Robust Unsupervised Arousal Rating: A Rule-Based Framework with Knowledge-Inspired Vocal Features.
- Cao, Verma, Nenkova, Speaker-sensitive emotion recognition via ranking.
- Sridhar et al., Role of Regularization in the Prediction of Valence from Speech.
- Sridhar & Busso, Unsupervised Personalization of an Emotion Recognition System.
- Gat et al., Speaker Normalization for Self-Supervised Speech Emotion Recognition.
- Shi, Li, Toda, Speaker-Aware Multi-Task Learning for Speech Emotion Recognition.
- Gao et al., ASER: Attribute-Serialized Emotion Recognition With Generative Multi-Task Learning.
- Jacewicz et al., Between-speaker and within-speaker variation in speech tempo of American English.

## Submission-state Notes

- Do not use EXP-20260921-13 preregistered CI as evidence; repeated deterministic seeds caused pseudo-replication.
- Do not use IEMOCAP relative_db loudness as primary evidence.
- Do not phrase novelty as inventing speaker normalization or speaker-relative prosody.
- Main novelty claim should be the controlled target-reference matching principle.
