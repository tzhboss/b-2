# Venue Strategy — 2026-09-21

## Current manuscript character

The paper is primarily:
- speech emotion recognition / computational paralinguistics;
- interpretable prosody;
- representation analysis;
- between-speaker versus within-speaker target structure;
- controlled evaluation methodology.

It is not primarily:
- database systems;
- information retrieval;
- a new large speech foundation model;
- a new SER SOTA architecture.

## COLING 2027

Current deadline:
- ARR October 2026 submission: 2026-10-12.
- Long papers: up to 8 main-body pages.
- Mandatory limitations section.
- Official topics include Phonology and Speech, Interpretability and Analysis, Machine Learning,
  Multimodality, and Sentiment Analysis.

Fit:
Plausible if framed as a computational-linguistic analysis of prosodic representation and target
semantics/reference frames.

Required framing changes:
1. Lead with the scientific question about what absolute versus speaker-relative prosody means.
2. Make the controlled target intervention the primary contribution.
3. Keep the categorical emotion/gender reversal as motivation, not the headline.
4. Emphasize analysis and evaluation methodology over raw SER performance.
5. Include a strong discussion of linguistic/prosodic between- and within-speaker variability.
6. Keep WavLM as representation-analysis evidence.
7. Explicitly separate population-level affect targets from person-relative affect targets.

Risk:
The work is still more speech/paralinguistic than typical text-centric COLING work. Reviewers may
ask for stronger links to spoken-language modeling or general representation learning.

## Interspeech 2027

Current paper deadline:
- 2027-02-09.

Fit:
Very strong topical fit. The conference covers all scientific and technological aspects of speech.

Best framing:
- Prosodic reference frames.
- Speaker normalization versus personalization.
- Continuous affect prediction.
- Interpretable low-dimensional prosody plus SSL representation analysis.

Advantages:
- Natural audience for Bone 2012, Busso 2013, speaker normalization, prosody, and SER.
- Less need to justify why the paper is about speech rather than text/NLP.
- The mechanism and careful negative-result audit are likely to be legible to reviewers.

Tradeoff:
The paper will need to be compressed heavily to the Interspeech paper format once the 2027
submission kit is final.

## IEEE Transactions on Affective Computing

Fit:
Very strong archival fit because the paper studies affect recognition, prosody, speaker
personalization/invariance, and a conceptual mechanism rather than only a benchmark.

Advantages:
- Enough space for the full experiment chain, negative results, and deployment analysis.
- The reference-frame principle and affect-target decomposition fit a theory/analysis-oriented
  affective-computing contribution.
- Prior closest works (Busso, Sridhar) are already part of this literature.

Tradeoff:
Longer journal review cycle; more pressure for complete literature coverage and potentially an
additional external dataset or stronger session-level validation.

## IEEE/ACM TASLP

Fit:
Strong if the manuscript emphasizes speech representation, prosody, normalization, and
speaker-dependent signal structure.

Advantages:
- Natural home for signal/representation analysis.
- Full-length journal space.
- Reproducibility emphasis matches the experiment registry and audit trail.

Tradeoff:
May favor stronger speech-processing methodology than the current deliberately simple
low-dimensional Ridge setup. The paper must argue that simplicity is necessary for causal
interpretability of the reference-frame intervention.

## DASFAA 2027

Current research-paper deadline:
- 2026-11-25.

Fit:
Weak.

DASFAA is centered on database systems, data management, data engineering, analytics, and advanced
applications. The current paper has no substantive database/data-management contribution.

Recommendation:
Do not reshape this paper solely to meet the DASFAA date. Doing so would weaken the natural story
and introduce avoidable scope risk.

## Practical submission paths

### Path A — Fast COLING attempt

Use the current manuscript as the source, compress to an 8-page ARR paper by 2026-10-12.

Main text:
1. Introduction
2. Related Work
3. Reference-Frame Formulation
4. Experimental Setup
5. Categorical motivation
6. Controlled MSP/IEMOCAP target intervention
7. Center permutation + K-shot
8. Discussion / Limitations

Move to appendix:
- full WavLM layer plots
- annotation disagreement
- speaker-ID analysis
- attribute-by-attribute secondary tables
- IEMOCAP loudness failure
- session-disjoint sensitivity details

### Path B — Interspeech 2027

Use the next months to tighten the paper, potentially add one more external continuous-affect
dataset if a legitimate one becomes available, then compress to the official Interspeech format.

### Path C — Journal-first

Develop the full mechanism paper for T-AFFC or TASLP.
Keep all audited positive and negative experiments and add deeper statistical treatment.

## Current recommendation

If schedule pressure matters, prepare a COLING 2027 ARR version now because the evidence is already
sufficient for a coherent analysis paper and the official scope includes speech/phonology and model
analysis.

In parallel, keep the full-length manuscript intact. If the COLING review outcome or scope fit is
weak, the same audited research package naturally supports an Interspeech 2027 or journal version.
