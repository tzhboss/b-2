# Experiment Audit — EXP-20260921-13

## Runtime integrity

- Source/runtime script and config hashes match.
- Immutable /tmp execution used.
- Session overlap is zero in every fold.
- Speaker overlap is zero in every fold.
- 10,039 utterances, 10 speakers, 5 sessions.

## Inferential validity failure

The leave-one-session-out folds are deterministic and Ridge is deterministic.
The implementation nevertheless repeated the same five folds for three seeds.

Audit confirms:
- maximum CCC spread across the three seeds: exactly 0;
- every seed×fold result is an exact duplicate of the corresponding fold.

Therefore treating 15 seed×fold rows as 15 independent bootstrap units is pseudo-replication.
The preregistered confidence intervals and significance criterion are invalid.

## Descriptive point estimates only

Mean Relative-minus-Absolute slope:
- Linear Arousal: -0.0124.
- Linear Dominance: -0.0051.
- Quadratic Arousal: -0.0173.
- Quadratic Dominance: -0.0110.

However, session-level slope signs are heterogeneous:
- Linear Arousal: 3/5 negative.
- Linear Dominance: 2/5 negative.
- Quadratic Arousal: 3/5 negative.
- Quadratic Dominance: 3/5 negative.

The lambda=0 Relative-minus-Absolute endpoint remains positive for Arousal/Dominance under both
models, but this experiment cannot provide valid confirmatory inference for the slope.

## Verdict

- Execution: completed.
- Validity: invalid for preregistered inferential claims.
- Decision: inconclusive.
- Keep only as a session-level sensitivity analysis after corrected aggregation.
