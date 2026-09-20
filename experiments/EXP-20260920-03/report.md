# Experiment Report — EXP-20260920-03

## Registered Hypothesis

The pitch reference-frame reversal observed in EXP-20260920-02 will remain detectable after
adding explicit prosody to frozen WavLM-large: WavLM plus speaker-relative pitch will outperform
WavLM plus absolute pitch for emotion, while WavLM plus absolute pitch will outperform WavLM
plus speaker-relative pitch for gender. The expected effect was registered as smaller than in
the prosody-only probe.

## Results

WavLM-only is already very strong:
- ESD emotion macro-F1: 0.9200.
- MEAD emotion macro-F1: 0.8245.
- RAVDESS emotion macro-F1: 0.8209.
- ESD gender macro-F1: 0.9997.
- MEAD gender macro-F1: 0.9996.
- RAVDESS gender macro-F1: 0.9974.

The key registered Relative-minus-Absolute pitch differences are tiny:
- ESD emotion: +0.00047, CI crosses zero.
- MEAD emotion: +0.00014, CI crosses zero.
- RAVDESS emotion: +0.00267, CI crosses zero.
- ESD gender: +0.00006.
- MEAD gender: +0.00009 with CI just above zero.
- RAVDESS gender: +0.00071, CI crosses zero.

Thus the prosody-only task reversal does not survive as a clear incremental Relative-vs-Absolute
effect when conditioning on final-layer mean-pooled WavLM-large.

Hybrid or multi-attribute explicit prosody can still add information. Relative to WavLM-only:
- ESD emotion + all-prosody Hybrid: +0.01473 macro-F1 [0.01344, 0.01588].
- ESD emotion + pitch Hybrid: +0.00738 [0.00651, 0.00831].
- MEAD emotion + all-prosody Hybrid: +0.00264 [0.00163, 0.00362].
- RAVDESS emotion + all-prosody Hybrid: +0.00498, CI crosses zero.
- Gender is nearly saturated under WavLM-only, leaving little meaningful headroom.

## Observation

The reference-frame effect is strongly representation-dependent. In the explicit prosody-only
probe, pitch shows a large task reversal. After adding final-layer frozen WavLM-large, Absolute
and Relative pitch produce nearly indistinguishable incremental performance. This is consistent
with, but does not prove, the interpretation that WavLM already represents both stable speaker
traits and within-speaker state cues.

The more robust positive signal in EXP-20260920-03 is not Relative versus Absolute alone; it is
that Hybrid/all-prosody explicit information can still improve emotion prediction on ESD and,
more modestly, MEAD beyond WavLM-only.

## Supported Claim

Under final-layer mean-pooled frozen WavLM-large and the registered known-speaker folds, the
large Absolute-versus-Relative pitch reversal observed in prosody-only probes is not detectable
as a comparable incremental effect. Explicit hybrid prosody nevertheless adds statistically
resolved emotion information on ESD and MEAD.

## Unsupported Stronger Claim

Do not claim that WavLM is universally invariant to prosodic reference frame, that explicit
relative prosody is unnecessary, or that final-layer mean pooling is optimal. This experiment
tests one WavLM checkpoint, one layer, one pooling scheme, one linear probe family, and controlled
known-speaker splits.

## Decision

inconclusive under the preregistered conditional task-reversal criterion.

## Next Step

Register a layer-wise WavLM experiment to test whether earlier/middle layers retain stronger
Absolute-versus-Relative structure than the final layer. Replace near-saturated gender with a
less trivial trait probe such as speaker identity or speaker-baseline prediction, and add
unseen-speaker/reference-estimation settings.
