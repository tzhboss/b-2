# Experiment Report — EXP-20260923-03

## Result

emotion2vec+ large was evaluated on all 10,039 IEMOCAP utterances with the same 5-fold speaker-disjoint Ridge protocol used for the WavLM gating experiment.

- Arousal: Overall 0.6228, Between 0.6253, Within 0.6227.
- Dominance: Overall 0.5053, Between 0.4909, Within 0.5060.

The Overall and Within scores are nearly identical for both targets. Against the existing panel, emotion2vec remains below WavLM and above Pitch+Rate for both Overall and Within, so it does not create the preregistered Overall-versus-Within ordering reversal.

Between-speaker intervals are very wide because IEMOCAP has only ten speakers; Between should therefore remain a descriptive diagnostic here rather than a strong inferential endpoint.

## Interpretation

Adding an emotion-specialized backbone does not rescue the strong Route-A rank-reversal hypothesis. Across Pitch+Rate, WavLM, and emotion2vec, conventional Overall and Within rankings are largely aligned. Route A should therefore be narrowed away from a generic claim that Overall ranking hides a different Within ranking.

The more defensible signal remains component attribution: some interventions (especially Hybrid in MSP and WavLM layer choice in IEMOCAP) change Between calibration much more than Within tracking. This can support a diagnostic decomposition claim, but it needs stronger robustness and intervention evidence rather than a benchmark-rank-reversal claim.

## Decision

weak_for_rank_reversal; informative_for_component_attribution.
