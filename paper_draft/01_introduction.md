# 1. Introduction

Speech emotion recognition (SER) models must separate affective variation from the many other
sources of acoustic variability carried by speech. Speaker identity is one of the most persistent
sources of such variation: speakers differ systematically in habitual pitch, loudness, speaking
rate, voice quality, accent, physiology, and expressive style. A common response is therefore to
normalize speaker-dependent variation or to learn speaker-invariant representations. Classical
feature-normalization approaches estimate speaker-specific reference statistics, often from neutral
speech, and transform expressive utterances relative to those references. More recent neural
approaches suppress speaker information adversarially or align speaker domains. Across these
lines of work, the usual premise is that speaker variability is primarily a nuisance that impairs
generalization.

That premise is useful, but incomplete. Emotion targets themselves can contain stable
between-speaker structure. Two speakers may differ not only in their acoustic baselines but also in
their population-level affect ratings, annotation distributions, or habitual expressive ranges. In
such a setting, removing all stable speaker information can discard signal that is predictive of the
target. Conversely, for a task that explicitly asks whether a speaker is above or below their own
typical state, stable speaker baselines should be nuisance by construction.

This distinction suggests that speaker normalization should be treated as a **reference-frame
choice**, not as a universally beneficial invariance operation. We write a simple acoustic
decomposition as

x_(s,u) = mu_s + delta_(s,u),

where mu_s is a stable speaker baseline and delta_(s,u) is an utterance-level deviation. A
speaker-relative representation suppresses mu_s and emphasizes delta_(s,u), whereas an absolute
representation retains both. A Hybrid representation can expose the deviation and baseline
separately. The target can be decomposed analogously:

y_(s,u) = ybar_s + epsilon_(s,u),

where ybar_s captures between-speaker target structure and epsilon_(s,u) captures within-speaker
variation. This leads to the central hypothesis of this work:

> **Representation reference frame should match target reference frame.**

The hypothesis predicts that speaker-relative prosody should be advantageous for within-speaker
targets, while absolute or Hybrid representations can become useful as the target contains more
between-speaker structure. Importantly, this is not a claim that speaker-relative prosody is always
better for emotion recognition.

We test this hypothesis through a sequence of controlled experiments. First, low-dimensional
prosody establishes an intuitive state-versus-trait reversal: speaker-relative pitch improves
categorical emotion classification while strongly reducing gender decoding across ESD, MEAD, and
RAVDESS. We then move to continuous valence-arousal-dominance (VAD) prediction. On MSP-Podcast,
raw Arousal and Dominance contain substantial between-speaker variance, and stable speaker acoustic
centers are strongly useful. Once the speaker mean is removed from the target, however,
speaker-relative prosody becomes preferable.

To move beyond post-hoc decomposition, we introduce a controlled target intervention. For each
speaker, we continuously vary the strength of the between-speaker target component while holding
utterances, acoustic inputs, folds, and model family fixed. With semantically aligned Pitch +
Speaking Rate features, the Relative-minus-Absolute performance difference decreases significantly
as between-speaker target strength increases in both MSP-Podcast and IEMOCAP, under both linear and
quadratic Ridge models. At the pure within-speaker endpoint, Relative outperforms Absolute for
Arousal and Dominance in every confirmatory corpus-by-model cell.

We further test the proposed mechanism rather than only its correlation. In a speaker-center
permutation intervention, the marginal distribution of speaker centers is preserved while center
identity is reassigned across speakers. On raw MSP Arousal and Dominance, using the correct center
substantially outperforms the permuted center; after the target is centered within speaker, that
difference nearly vanishes. Finally, we evaluate practical reference estimation with label-free
speaker enrollment and find that approximately 20 enrollment utterances recover most of the usable
speaker-reference benefit on a fixed MSP downstream pool.

These results support a conditional view of speaker normalization. Stable speaker information can
act as nuisance or useful information depending on the structure of the target. This framing helps
reconcile two active directions in SER: methods that suppress speaker information to improve
speaker-independent robustness, and personalization methods that explicitly exploit speaker
characteristics. Rather than treating one direction as universally correct, our experiments show
that their usefulness depends on which reference frame the task is asking the model to predict.

## Contributions

1. We formulate prosodic normalization and target centering within a common reference-frame
   analysis, explicitly connecting two established between/within-speaker decompositions.
2. We introduce a controlled target-reference intervention that continuously varies
   between-speaker target strength without changing acoustic inputs or model architecture.
3. We provide confirmatory cross-corpus evidence on MSP-Podcast and IEMOCAP, using semantically
   matched Pitch + Speaking Rate features and both linear and quadratic regression models.
4. We provide a speaker-center identity intervention showing that stable speaker baselines help
   only when they are correctly aligned to the speaker and the target contains speaker-level
   structure.
5. We evaluate a label-free K-shot deployment setting and quantify how much unlabeled speaker
   reference audio is needed to recover practical baseline utility.
6. We characterize important boundaries: Valence is weak for this low-dimensional prosody
   mechanism; IEMOCAP loudness provenance is not robust enough for the main claim; and strict
   session-disjoint IEMOCAP effects are heterogeneous across only five sessions.
