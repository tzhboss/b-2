# Experiment Audit — EXP-20260921-08

## Runtime integrity and fixed-pool control

- Source/runtime script and config SHA256 values match.
- Immutable /tmp execution used.
- IEMOCAP: 10,039 utterances, 10 speakers.
- Exactly 100 enrollment utterances reserved per speaker.
- Downstream rows per seed: 9,039.
- Train/test row hashes are identical across all K within every seed/fold.
- No metric NaNs.

## Reference convergence

Mean center MAE to full-speaker oracle:

Pitch:
- K5 1.9068
- K100 0.3970

Loudness:
- K5 2.1184
- K100 0.3773

Log-rate:
- K5 0.1738
- K100 0.0461

K100 reduces MAE by more than 50% versus K5 for all three attributes.

## Target-reference slopes

Oracle Relative-minus-Absolute:
- Arousal: -0.03418, 95% CI [-0.04921, -0.02054].
- Dominance: -0.02292, [-0.03045, -0.01562].

K-shot slopes are not monotonic with K.

Arousal:
- K5 -0.00033
- K10 -0.01940
- K20 -0.01079
- K50 -0.02531, CI below zero
- K100 -0.01819, CI below zero

Dominance:
- K5 +0.00138
- K10 -0.03513, CI below zero
- K20 -0.00843
- K50 -0.01397, CI below zero
- K100 -0.00780, CI crosses zero

K100 is closer to oracle than K5 for both targets, but it is not the closest K; K50 is closer for
both Arousal and Dominance.

## Registered Criteria

- Reference convergence: supported.
- K100 within 0.01 of oracle slope: not supported.
- K100 significant negative coupling for both Arousal and Dominance: not supported.
- K100 gap smaller than K5 gap: supported.
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: mixed.
- Better center estimation reduces average oracle gap, but mechanism recovery is not monotonic in K.
