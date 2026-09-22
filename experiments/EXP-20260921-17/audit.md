# Experiment Audit — EXP-20260921-17

## Runtime integrity

- Restored and executed the original preregistered EXP-20260921-17 definition.
- Solver: LSQR.
- Ridge alpha: 1.0.
- LSQR tolerance: 1e-8.
- Fixed speaker-disjoint split seed: 20260920.
- Speaker-cluster bootstrap: 5,000 replicates.
- Runtime script/config SHA256 values match source.
- Completed with exit code 0.

A duplicate registration attempt using the same experiment ID was detected before scientific audit,
stopped, archived, and the original stricter preregistration restored. The archived duplicate run
is not used as scientific evidence.

## LSQR results

Layer 12:
- Arousal slope: -0.082938, 95% CI [-0.130916, -0.035905].
- Dominance slope: -0.034061, [-0.058346, -0.011798].
- Arousal lambda=0 effect: +0.040463.
- Dominance lambda=0 effect: +0.014833.

Layer 24:
- Arousal slope: -0.095777, 95% CI [-0.142897, -0.048177].
- Dominance slope: -0.045550, [-0.070665, -0.018104].
- Arousal lambda=0 effect: +0.046590.
- Dominance lambda=0 effect: +0.025066.

All four slopes remain negative with CIs below zero and all four lambda=0 effects remain positive.

## Quantitative comparison with EXP-20260921-16 default solver

Maximum absolute slope difference:
- 1.65e-6.

Maximum absolute lambda=0 endpoint difference:
- 1.01e-6.

Both are orders of magnitude smaller than the preregistered <=0.01 stability threshold.

## Registered criteria

- All slope signs preserved: supported.
- All lambda=0 signs preserved: supported.
- All slope differences <=0.01: supported.
- All endpoint differences <=0.01: supported.
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: pass.

The WavLM target-reference effect is numerically stable to replacing the default Ridge solve with a
high-precision LSQR solve.
