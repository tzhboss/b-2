# Prosodic Reference Frames — Paper Evidence Synthesis

## Core thesis

Prosodic normalization is an information transformation rather than a universally beneficial preprocessing step. The preferred acoustic reference frame depends on the reference frame of the prediction target.

A compact decomposition is x_su = μ_s + δ_su and y_su = speaker_mean_s + ε_su.

- Absolute prosody retains stable speaker baseline plus utterance deviation.
- Relative prosody emphasizes within-speaker deviation.
- Hybrid exposes both components.
- A target dominated by within-speaker state should prefer Relative.
- A target containing stable between-speaker structure can benefit from Absolute/Hybrid.

## Main-paper experiment stack

1. Categorical state/trait reversal — EXP-20260920-02.
2. Target decomposition — EXP-20260920-17.
3. Semantically matched cross-corpus intervention — EXP-20260921-12.
4. Speaker-center permutation — EXP-20260921-06.
5. K-shot deployment — EXP-20260920-21.
6. WavLM representation analysis — EXP-20260920-04.

## Claims that should be constrained

- Do not claim Relative normalization universally improves emotion prediction.
- Do not claim speaker identity is removed by Relative prosody; EXP-20260921-11 shows only modest suppression.
- Do not use IEMOCAP relative_db as primary loudness evidence; EXP-20260921-09 rejected semantic robustness.
- Do not claim monotonic K-shot mechanism recovery on IEMOCAP; EXP-20260921-08 found non-monotonic slope recovery.
- Treat Valence as a weak/negative-control dimension for the low-dimensional prosody story.

## Recommended main figures

- Fig. 1: fig1_target_reference_matching_pitch_rate.png — primary mechanism curve.
- Fig. 2: fig2_kshot_reference_saturation_msp.png — deployment practicality.
- Fig. 3: fig3_emotion_gender_pitch_reversal.png — intuitive state-vs-trait motivation.
- Fig. 4: fig4_wavlm_reference_decodability_esd.png — representation analysis, optional main/appendix depending space.

## Recommended paper framing

Working claim: Representation reference frame should match target reference frame.

This is stronger and more defensible than saying speaker-relative prosody is always better, because it explains both wins and failures under one mechanism.

## Session-disjoint IEMOCAP sensitivity

A stricter leave-one-session-out analysis was attempted in EXP-20260921-13. The original
inferential implementation was invalid because the same deterministic five session folds were
duplicated under three seed labels. After collapsing to the five unique sessions, mean
Arousal/Dominance slopes remain negative, but only 2-3 of 5 sessions show negative slopes per
model-target cell. This should be reported as substantial session heterogeneity and a limitation,
not as confirmatory evidence.

## Extended attribute validation — EXP-20260921-14

The reference-frame effect is not confined to pitch level, loudness, or speaking rate.

On MSP, Relative-minus-Absolute slope versus between-speaker target strength is significantly
negative for:
- F0 variability: Arousal -0.1053; Dominance -0.0789.
- Pause ratio: Arousal -0.0736; Dominance -0.0593.
- Voiced ratio: Arousal -0.1598; Dominance -0.1169.
- Combined variability/temporal set: Arousal -0.2616; Dominance -0.1977.

At the pure within-speaker endpoint, the combined set still favors Relative:
+0.0363 CCC for Arousal and +0.0274 for Dominance.

This broadens the mechanism from acoustic level features to expressive range and temporal/voicing
organization.

## Corrected independent-unit inference — EXP-20260921-15

The preferred uncertainty analysis now treats speaker as the independent sampling unit.

MSP speaker-cluster bootstrap:
- Linear Arousal slope -0.3188, 95% CI [-0.3384, -0.2998].
- Linear Dominance -0.1825, [-0.1969, -0.1683].
- Quadratic Arousal -0.3214, [-0.3411, -0.3022].
- Quadratic Dominance -0.1851, [-0.2000, -0.1705].

All four MSP slope intervals remain strictly below zero.

IEMOCAP point slopes remain negative in all four model-target cells, but cluster-bootstrap CIs cross
zero because only 10 speakers are available. Its pure within-speaker endpoint advantage remains
positive with CIs above zero.

Therefore:
- strong inferential support comes from MSP;
- IEMOCAP contributes directional external replication and a robust within-speaker endpoint, but
  should not be described as independently significant for the slope under speaker-cluster
  inference.

## Learned representation validation — EXP-20260921-16

The main mechanism also appears when reference frames are constructed directly in frozen WavLM
embedding space rather than from handcrafted prosodic attributes.

Speaker-cluster Relative-minus-Absolute slopes:

- Layer 12 Arousal: -0.0829, 95% CI [-0.1309, -0.0359].
- Layer 12 Dominance: -0.0341, [-0.0583, -0.0118].
- Layer 24 Arousal: -0.0958, [-0.1429, -0.0482].
- Layer 24 Dominance: -0.0455, [-0.0707, -0.0181].

All four slope intervals are below zero.

At lambda=0, Relative WavLM embeddings are significantly better in all four cells. At lambda=1,
Absolute embeddings are better in all four cells. The reference preference therefore crosses zero
as the target changes from within-speaker to population-level.

This materially reduces the risk that the paper is merely documenting an arithmetic property of
handcrafted pitch/rate normalization.


## Numerical solver stability — EXP-20260921-17

The WavLM intervention is numerically stable. Replacing the default Ridge solver with LSQR at
tolerance 1e-8 changes the four primary slopes by at most 1.65e-6 and the lambda=0 endpoint effects
by at most 1.01e-6. All signs and confidence conclusions are unchanged.

This removes the ill-conditioned-matrix warning as a plausible explanation for the WavLM
reference-frame effect.
