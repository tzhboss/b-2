# Suggested Paper Structure

## 1. Introduction

Problem:
Speaker normalization is usually treated as a robustness operation, implicitly assuming stable
speaker variation is nuisance.

Counterpoint:
The same stable speaker baseline may be useful when the prediction target itself contains
between-speaker structure.

Core claim:
Representation reference frame should match target reference frame.

## 2. Reference-Frame Formulation

Acoustic decomposition:
x_su = μ_s + δ_su

Target decomposition:
y_su = ȳ_s + ε_su

Representations:
- Absolute: retains μ_s + δ_su
- Relative: emphasizes δ_su
- Hybrid: exposes δ_su and μ_s separately

Prediction:
- within-speaker targets should prefer Relative
- targets containing between-speaker structure can use Absolute/Hybrid

## 3. Experimental Setup

Datasets:
- ESD, MEAD, RAVDESS for categorical state/trait motivation
- MSP-Podcast for large-scale continuous VAD
- IEMOCAP for external continuous-VAD validation

Primary acoustic attributes:
- pitch
- speaking rate
- MSP loudness as additional attribute analysis

Primary metrics:
- Macro-F1 for categorical tasks
- speaker-balanced CCC for VAD

## 4. State-versus-Trait Motivation

EXP-20260920-02:
Relative pitch helps categorical emotion but strongly hurts gender decoding.

Use this as motivation, not the final mechanism claim.

## 5. Target Reference Frame Explains the Reversal

EXP-20260920-17:
MSP VAD target decomposition.

EXP-20260921-12:
Main confirmatory lambda intervention with semantically matched Pitch+Rate across MSP and IEMOCAP,
linear and quadratic Ridge.

This should be the central results section.

## 6. Why Speaker Center Helps

EXP-20260921-06:
True-center versus speaker-permuted-center intervention.

Main point:
The correct center identity matters strongly for raw MSP Arousal/Dominance but not for centered
within-speaker targets.

## 7. Practical Reference Estimation

EXP-20260920-19 and EXP-20260920-21:
Label-free K-shot enrollment.

Use EXP-21 as the corrected fixed-downstream-pool result.
Approximately K=20 recovers most practical utility on MSP.

## 8. Representation-Level Analysis

EXP-20260920-04:
WavLM decodes Absolute, Relative, and implied baseline pitch across layers.

Constrain claim:
WavLM preserves multiple frames; explicit Relative augmentation does not universally improve
emotion after WavLM.

## 9. Robustness and Boundaries

Include:
- model-family robustness
- annotation-disagreement robustness
- attribute decomposition
- IEMOCAP loudness provenance failure
- session-disjoint IEMOCAP sensitivity and limited session count
- speaker-ID decodability constraint

## 10. Discussion

Main interpretation:
Normalization redistributes information rather than universally removing noise.

Implications:
- SER evaluation should specify target reference frame.
- Speaker-invariant representations may discard task-relevant population-level affect priors.
- Within-speaker monitoring tasks may benefit from relative representations.
- Deployment requires explicit reference-estimation assumptions.

## Recommended main-text experiment IDs

- EXP-20260920-02
- EXP-20260920-17
- EXP-20260921-12
- EXP-20260921-06
- EXP-20260920-21

Optional main / appendix:
- EXP-20260920-04
- EXP-20260921-04
- EXP-20260921-05
- EXP-20260920-18
- EXP-20260921-11

Sensitivity / limitations:
- EXP-20260921-08
- EXP-20260921-09
- EXP-20260921-13
