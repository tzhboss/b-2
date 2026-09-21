# Novelty Statement

## One-sentence version

Prior work has independently studied speaker-relative normalization for emotion recognition and
within-person centering for affect analysis; we connect these two reference frames experimentally,
showing that the utility of Absolute versus Relative prosody changes systematically when the
between-speaker component of the prediction target is manipulated.

## COLING-style paragraph

Speaker normalization and person-mean centering are not new individually. Prior SER work has used
speaker-specific neutral baselines to construct relative prosodic arousal measures and to suppress
inter-speaker variability, while multilevel affect research routinely separates within-person from
between-person variation by centering. Our contribution is to treat these as two sides of the same
reference-frame problem. We intervene on the target by continuously varying its between-speaker
component while keeping utterances, acoustic inputs, splits, and model family fixed, and test
whether the preferred acoustic reference frame changes accordingly. The resulting
Relative-minus-Absolute slope replicates across MSP-Podcast and IEMOCAP with semantically matched
Pitch + Speaking Rate and two model families, and a speaker-center permutation intervention shows
that raw-target baseline utility depends on assigning the correct center to the correct speaker.

## Rebuttal-safe boundary

We do not claim:
- the first use of speaker normalization in SER;
- the first use of relative prosody for arousal;
- the first use of speaker enrollment;
- the first decomposition of within- and between-person effects;
- the first person-mean centering of affect measures.

The candidate novelty is the controlled **representation-reference × target-reference matching**
hypothesis and its intervention-based validation.
