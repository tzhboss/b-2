# Candidate Titles and Abstract

## Recommended title

**Normalization Is Not Neutral: Matching Prosodic Reference Frames to Affective Targets**

Why this is the best default:
- states the conceptual contribution rather than a specific model;
- makes the non-universal nature of normalization explicit;
- naturally covers categorical emotion, continuous VAD, K-shot deployment, and SSL analysis.

## Alternative titles

1. **Prosodic Reference Frames: When Speaker Normalization Helps—and When It Removes Useful Affect Information**
2. **When Is Speaker Variation Nuisance? Target-Dependent Prosodic Reference Frames for Speech Affect**
3. **Absolute or Speaker-Relative? A Reference-Frame View of Prosody for Emotion Recognition**
4. **Speaker Baseline as Nuisance or Signal: Matching Prosodic Representations to Affective Targets**
5. **Target Reference Frames Govern the Utility of Speaker-Normalized Prosody**
6. **Beyond Speaker Normalization: Reference-Frame Matching for Speech Emotion Recognition**

## Abstract — conference version

Speaker normalization is widely used in speech emotion recognition because prosodic measurements
vary strongly across speakers. This practice often treats stable speaker variation as nuisance.
We instead ask when speaker baseline information should be removed and when it should be preserved.

We decompose utterance-level representations and continuous affect targets into between-speaker and
within-speaker components, yielding Absolute, speaker-Relative, and Hybrid reference frames.
Across categorical emotion tasks, Relative pitch improves emotion recognition while suppressing
stable speaker-trait information. On MSP-Podcast, speaker-cluster inference over 1,915 speakers
shows that Relative-minus-Absolute Arousal and Dominance CCC decreases strongly as the
between-speaker component of the target is increased. On IEMOCAP, the same handcrafted
Pitch+Speaking-Rate slopes remain directionally negative but are imprecise with only ten speakers,
while the pure within-speaker endpoint robustly favors Relative representations.

We further repeat the intervention directly in frozen WavLM-large embedding space. At both layer 12
and layer 24, Arousal and Dominance show significantly negative target-reference slopes under
speaker-cluster bootstrap; Relative embeddings are favored for pure within-speaker targets and the
preference reverses for raw targets. A speaker-center permutation experiment shows that raw MSP
Arousal/Dominance gains require the correct speaker center, while center identity becomes nearly
irrelevant after target centering. Additional experiments extend the effect to F0 variability,
pause ratio, and voiced ratio, and label-free K-shot enrollment recovers most practical
speaker-reference utility on MSP.

These results support a simple principle: **representation reference frame should match target
reference frame**. Speaker normalization is therefore better viewed as an information
transformation than as a universally beneficial invariance operation.

## Abstract — shorter version

Speaker normalization in speech emotion recognition usually assumes that stable speaker variation
is nuisance. We test a different hypothesis: whether speaker baseline information should be
removed or preserved depends on the reference frame of the target.

We decompose both prosodic features and continuous affect targets into between-speaker and
within-speaker components, yielding Absolute, Relative, and Hybrid prosodic representations.
Across categorical emotion tasks, Relative pitch improves emotion recognition while reducing
stable speaker-trait information. More importantly, we directly manipulate the between-speaker
component of Arousal and Dominance targets. With semantically matched pitch and speaking rate,
Relative-minus-Absolute CCC decreases significantly as between-speaker target strength increases
in both MSP-Podcast and IEMOCAP under linear and quadratic models. At the pure within-speaker
endpoint, Relative is consistently superior.

Permuting speaker centers shows that Hybrid gains on raw MSP targets require the correct speaker
identity, while the same center information becomes irrelevant after within-speaker target
centering. Label-free K-shot enrollment further recovers most practical center utility with about
20 utterances.

Our results suggest that speaker normalization should be viewed as a reference-frame choice:
**representation reference frame should match target reference frame.**
