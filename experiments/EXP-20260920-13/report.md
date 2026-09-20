# Experiment Report — EXP-20260920-13

Training-only confidence selection is too conservative. It reduces harmful switches slightly in
some corpora but sacrifices the large adaptive gains available in MSP, lowering aggregate
performance relative to the naive 0.5 stats-only router.

This negative result is useful: the practical routing benefit in EXP-12 comes from committing to
cell-specific switches rather than abstaining toward the globally preferred Relative policy.

Decision: reject.
