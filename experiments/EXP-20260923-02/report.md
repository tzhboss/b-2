# Experiment Report — EXP-20260923-02

## Result

All four models were evaluated on the exact same 10,039 IEMOCAP utterances for Arousal and
Dominance at the unchanged absolute-label endpoint.

The preregistered strong Route-A result was **not observed**: no pairwise model ordering reversed
between Overall CCC and Within-speaker CCC.

Arousal:
- Overall: WavLM-L12 0.7210 > WavLM-L24 0.7091 > Pitch+Rate quadratic 0.3120 > linear 0.2979.
- Within: WavLM-L12 0.7131 > WavLM-L24 0.6973 > quadratic 0.3433 > linear 0.3337.
- Between differs: WavLM-L24 0.8881 > WavLM-L12 0.8485, while Pitch+Rate has approximately zero
  speaker-mean calibration.

Dominance:
- Overall and Within rankings are both WavLM-L24 ≈ WavLM-L12 >> Pitch+Rate quadratic > linear.
- WavLM-L24 has materially stronger Between point performance than L12 (0.5323 vs 0.3238), though
  only ten speakers make these Between intervals wide.

## Interpretation

The experiment does not support the stronger claim that standard Overall rankings are generally
reversed by a Within diagnostic. It does show that models with similar Overall/Within behavior can
have materially different Between-speaker calibration, especially across WavLM layers.

This is informative but insufficient by itself to justify a benchmark-redefinition paper. The next
gating step is to add an emotion-specialized representation (emotion2vec) on the same IEMOCAP
utterances. If broader model families still preserve essentially identical Overall and Within
rankings, Route A should be narrowed to component attribution rather than rank-reversal claims.

## Decision

informative_mixed.
