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
