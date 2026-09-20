# CURRENT_STATE.md

## Phase

Per-emotion and conflict-set reference-frame analysis completed and audited.

## Main valid findings

- EXP-02: explicit Relative pitch improves aggregate emotion on ESD, MEAD, and RAVDESS.
- EXP-03: final-layer WavLM largely absorbs the explicit Absolute-vs-Relative difference.
- EXP-04: WavLM retains strongly decodable Absolute, Relative, and speaker-baseline pitch information with clear layer dependence.
- EXP-05: on unseen speakers, K=10 label-free reference audio recovers oracle-level Relative-pitch emotion performance.
- EXP-06: label-free enrollment converges toward a speaker marginal pitch center rather than a privileged neutral reference.
- EXP-07: natural corpora are heterogeneous; MELD is near-neutral while MSP reverses and prefers Absolute in aggregate.
- EXP-08: MSP aggregate reversal is substantially explained by useful stable speaker-baseline information.
- EXP-09: the MSP reversal is class-localized, dominated by neutral; robust Absolute/Relative conflict examples are common and flip preferred reference frame between controlled and natural corpora.

## Paper-level interpretation

Prosodic reference frames redistribute information rather than universally improve representation.
The useful frame depends on task, corpus/domain, and emotion class. In MSP, Relative pitch is
better for most emotion classes, but neutral strongly benefits from stable speaker baseline
information, which is enough to reverse the aggregate result. On explicit reference-frame conflict
examples, controlled corpora favor Relative whereas MSP and MELD favor Absolute.

## Next legal step

Test whether the class-specific pattern generalizes beyond pitch by repeating the decomposition
for loudness and speaking rate, with preregistered per-emotion analyses and no new model tuning.
