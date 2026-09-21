# COLING 2027 Eight-Page Compression Plan

## Page-budget principle

The paper cannot carry every experiment in the main body.
The main body should prove one idea:

**The preferred prosodic reference frame changes with the reference frame of the target.**

## Main figures

Figure 1:
Concept diagram of x = speaker baseline + within-speaker deviation and
y = speaker mean + within-speaker deviation.

Figure 2:
Semantically matched MSP/IEMOCAP Pitch+Rate lambda curves (current paper Fig. 1).

Figure 3:
Speaker-center permutation on raw versus residual MSP targets.

Figure 4:
K-shot MSP enrollment curve, only if space permits.

## Main tables

Table 1:
Datasets and exact roles:
- ESD/MEAD/RAVDESS: categorical motivation.
- MSP: mechanism/deployment.
- IEMOCAP: external confirmation.

Table 2:
Confirmatory Pitch+Rate slopes with 95% CIs for MSP/IEMOCAP × Linear/Quadratic × A/D.

## Main-text result order

### R1 — Motivation
Pitch Relative-minus-Absolute improves emotion but hurts gender across all three categorical
corpora.

Use only one compact table or mini-panel.

### R2 — Target decomposition
MSP A/D have substantial between-speaker structure.
Within-speaker residual targets strongly favor Relative.

### R3 — Primary confirmation
Lambda intervention with matched Pitch+Rate:
all eight A/D corpus×model slopes are negative with CIs below zero.

This receives the most space.

### R4 — Mechanism
Correct speaker-center identity matters for raw MSP A/D but not residual targets.

### R5 — Deployment
K=20 unlabeled enrollment recovers most of the oracle utility.

## Appendix

Move:
- all individual attribute analyses;
- annotation disagreement;
- full WavLM layer analysis;
- WavLM conditional augmentation null results;
- speaker-ID decoding;
- IEMOCAP loudness semantic failure;
- session-disjoint sensitivity;
- all preregistration/audit details;
- full K curves and center MAE.

## COLING-specific wording

Prefer:
- representation analysis;
- spoken-language/prosodic representation;
- linguistic variability;
- evaluation target semantics;
- population-relative vs person-relative labels.

Avoid making the paper sound like:
- a feature-engineering SER system;
- a simple Ridge benchmark;
- a new normalization algorithm.

The algorithm is deliberately simple because the contribution is the controlled analysis.
