# CURRENT_STATE.md

## Phase

Controlled, WavLM, unseen-speaker, natural-corpus, and MSP mechanism experiments completed and audited.

## Main valid findings

- EXP-02: explicit Relative pitch improves emotion on ESD, MEAD, and RAVDESS while Absolute retains more trait information.
- EXP-03: final-layer WavLM largely absorbs the explicit Absolute-vs-Relative difference.
- EXP-04: WavLM retains strongly decodable Absolute, Relative, and speaker-baseline pitch information with clear layer dependence.
- EXP-05: on unseen speakers, K=10 label-free reference audio recovers oracle-level Relative-pitch emotion performance.
- EXP-06: label-free enrollment converges toward a speaker marginal pitch center rather than a privileged neutral reference; marginal reference can outperform neutral.
- EXP-07: natural corpora are heterogeneous; MELD is near-neutral while MSP reverses and prefers Absolute.
- EXP-08: MSP reversal is explained in substantial part by useful emotion information in the stable speaker pitch baseline removed by normalization.

## Current paper-level interpretation

Prosodic reference frames redistribute information rather than universally improve representation.
Relative pitch emphasizes within-speaker state, while stable speaker baselines can remain task-relevant.
The optimal decomposition depends on corpus/domain, and learned WavLM representations retain both components internally.

## Next legal step

Run per-emotion and conflict-set analysis to identify which emotion labels drive the corpus-dependent
baseline/relative trade-off, especially MSP versus ESD/RAVDESS.
