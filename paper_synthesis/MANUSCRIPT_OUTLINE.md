# Manuscript Outline — Prosodic Reference Frames

## Working title options

1. Prosodic Reference Frames: When Speaker Normalization Helps and When It Removes Useful Information
2. Normalization Is Not Neutral: Matching Prosodic Representations to Target Reference Frames
3. Absolute or Relative? Target Structure Determines the Utility of Speaker-Normalized Prosody

## One-sentence thesis

Speaker-relative prosody is not universally better: it is most useful when the prediction target is
defined within speaker, whereas stable speaker baselines become useful when the target itself
contains between-speaker structure.

## Research questions

### RQ1 — Is the utility of Absolute versus Relative prosody task-dependent?

Use explicit low-dimensional prosody to establish the phenomenon before introducing mechanism.

Main evidence:
- EXP-20260920-02.
- Relative pitch improves categorical emotion Macro-F1 across ESD/MEAD/RAVDESS.
- Relative pitch strongly harms gender prediction.
- Speaker-ID follow-up EXP-20260921-11 provides a conservative boundary: identity suppression is
  consistent but modest.

Main-paper placement:
- Table 1: Relative-minus-Absolute for emotion and gender, pitch/all.
- Short paragraph on speaker-ID boundary.

Do not overclaim:
- Gender is a strong trait example, not proof of universal identity removal.

### RQ2 — What determines which prosodic reference frame is useful?

This is the core contribution.

Mechanism:
- Decompose target into between-speaker and within-speaker components.
- MSP raw Arousal/Dominance contain large between-speaker structure.
- IEMOCAP contains much less.
- Introduce continuous target intervention:
  target(lambda) = global + within + lambda * between.

Primary evidence:
- EXP-20260920-17 target decomposition.
- EXP-20260921-02 cross-corpus contrast.
- EXP-20260921-03 controlled lambda intervention.
- EXP-20260921-04 linear/quadratic robustness.
- EXP-20260921-06 correct-center permutation.

Main-paper figures:
- Figure 1: Relative-minus-Absolute versus lambda for MSP/IEMOCAP.
- Figure 2: slope robustness across corpus/model family.
- Figure 3: correct versus permuted speaker center on raw/residual VAD.

Core claim:
- Relative-minus-Absolute utility decreases as between-speaker target structure increases.
- Correct speaker center helps only when the target retains speaker-level structure.
- This supports target-reference matching rather than a universal normalization advantage.

### RQ3 — Does the mechanism survive realistic reference estimation?

Deployment extension.

Evidence:
- EXP-20260920-19 K-shot raw-VAD recovery.
- EXP-20260921-01 K-shot saturation.
- EXP-20260921-07 deployment-style target intervention.
- EXP-20260921-08 IEMOCAP reference-quality curve.

Main-paper result:
- K-shot Hybrid can recover useful speaker reference without emotion or neutral labels.
- MSP reaches strong diminishing returns around K=20.

Main-paper figure:
- Figure 4: K-shot Hybrid CCC versus K.

Boundary:
- IEMOCAP K-shot slope recovery is directionally consistent but non-monotonic and noisier.

## Section structure

### 1. Introduction

Motivation:
- Speaker normalization is usually treated as benign nuisance removal.
- But prosodic observations contain both stable speaker baselines and dynamic state deviations.
- Removing the baseline changes the information available to the task.

Gap:
- Prior work often asks whether normalization improves performance.
- This paper asks what information normalization preserves/removes, and when that information is
  task-relevant.

Contributions:
1. Demonstrate task-dependent Absolute/Relative reversals across multiple speech corpora.
2. Introduce target-reference matching as a mechanism.
3. Validate the mechanism through target intervention, model robustness, and center permutation.
4. Show label-free K-shot reference estimation can recover useful speaker baselines.

### 2. Reference-frame formulation

Acoustic decomposition:
x_su = mu_s + delta_su + epsilon_su.

Target decomposition:
y_su = ybar + within_su + between_s.

Representations:
- Absolute: preserves both stable and dynamic components.
- Relative: suppresses stable speaker baseline.
- Hybrid: exposes deviation and baseline separately.

Prediction:
- within-speaker targets should favor Relative.
- targets with between-speaker structure should benefit from baseline information.

### 3. Experimental setup

Corpora:
- ESD, MEAD, RAVDESS for controlled categorical-emotion / gender analysis.
- MSP-Podcast for natural VAD and deployment experiments.
- IEMOCAP for external continuous-VAD validation.

Prosody:
- pitch
- loudness
- speaking rate

Important semantic restriction:
- IEMOCAP main evidence must use semantically safe pitch+rate.
- Do not use source relative_db as main loudness evidence.

Metrics:
- Macro-F1 for classification.
- speaker-balanced CCC for continuous VAD.

Splits:
- speaker-disjoint for affect/VAD.
- content-disjoint for speaker-ID representation audit.

### 4. RQ1: Reference-frame utility is task-dependent

Main table:
- EXP-02 controlled emotion versus gender.

Secondary:
- per-emotion / per-attribute heterogeneity from EXP-09/10.

Interpretation:
- normalization redistributes information rather than universally improving representation quality.

### 5. RQ2: Target-reference matching explains the reversal

#### 5.1 Raw VAD target structure

- MSP ICC-like Arousal about 0.42, Dominance about 0.35.
- IEMOCAP Arousal about 0.045, Dominance about 0.051.

#### 5.2 Controlled target intervention

- lambda from 0 to 1.
- MSP shows full sign crossover.
- IEMOCAP shows weaker but significant negative slopes.

#### 5.3 Model robustness

- Linear versus Quadratic Ridge.
- Preserve negative Arousal/Dominance slopes in both corpora.

#### 5.4 Correct speaker center is causal to the utility

- Speaker-center permutation.
- MSP raw Arousal/Dominance collapse with mismatched centers.
- Residual targets are essentially unchanged.

### 6. RQ3: Label-free reference estimation

- K-shot center estimation.
- K=10 already close to oracle.
- K around 20 near downstream practical plateau on MSP.
- Distinguish acoustic-center estimation accuracy from downstream utility.

### 7. Representation analysis

Keep short in main paper or move to appendix depending venue.

WavLM:
- absolute, relative, and implied baseline pitch are all highly decodable.
- reference information is reorganized with depth.
- final-layer WavLM does not show the explicit prosody Relative-versus-Absolute reversal.

Interpretation:
- learned representations can retain multiple reference frames simultaneously.
- explicit normalization changes task access more directly than frozen final-layer WavLM probing.

### 8. Discussion

Key message:
- normalization should be chosen relative to the target semantics.
- speaker baseline can be nuisance for within-speaker state prediction and useful signal for
  population-level targets.

Categorical versus dimensional affect:
- controlled emotion labels tend to reward within-speaker deviations.
- natural raw VAD can include stable between-speaker priors.
- therefore categorical and dimensional results need not agree.

Limitations:
- Valence is weakly predicted by these low-dimensional prosodic features.
- IEMOCAP has only 10 speakers.
- IEMOCAP loudness semantics are unresolved.
- some categorical per-class effects depend on classifier family.
- speaker-ID suppression is modest beyond gender.
- target intervention is diagnostic, not a claim that real-world labels are literally generated by
  the lambda interpolation.

### 9. Conclusion

Prosodic normalization is an information transformation. The best reference frame depends on the
reference frame of the target.

## Main-paper result budget

### Must include

1. EXP-02 task reversal.
2. MSP/IEMOCAP target-structure contrast.
3. Controlled lambda intervention.
4. Linear/quadratic robustness.
5. Speaker-center permutation.
6. One K-shot deployment curve.
7. Semantically safe IEMOCAP pitch+rate external replication.

### Appendix

- Per-emotion categorical breakdown.
- Full attribute-specific intervention.
- WavLM layer-wise decodability.
- Annotator-disagreement analysis.
- Full K-shot center-error tables.
- IEMOCAP K-shot non-monotonicity.
- speaker-ID content-disjoint audit.
- classifier-family sensitivity.
- failed/rejected experiments and provenance audit.

### Exclude from evidence

- EXP-20260920-01.
- EXP-20260920-20.
- IEMOCAP relative_db loudness claims as main evidence.

## Suggested main figures

Figure 1 — Conceptual reference-frame diagram.
- Absolute = speaker baseline + deviation.
- Relative = deviation.
- Hybrid = deviation plus explicit baseline.
- Target has within and between components.

Figure 2 — Controlled target intervention.
- MSP/IEMOCAP Arousal and Dominance.
- x-axis lambda.
- y-axis Relative-minus-Absolute CCC.

Figure 3 — Mechanism permutation.
- true center minus permuted center.
- raw versus residual targets.

Figure 4 — K-shot deployment.
- K versus CCC and/or oracle gap.

## Suggested main tables

Table 1 — Controlled task reversal.
Table 2 — Target ICC plus intervention slope.
Table 3 — Model-family robustness.
Table 4 — Semantically safe external replication.

