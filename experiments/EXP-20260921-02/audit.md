# Experiment Audit — EXP-20260921-02

## Runtime integrity

- Source/runtime script SHA256 match.
- Source/runtime config SHA256 match.
- Execution used immutable /tmp copies.
- Output schema matches preregistration.

## Completeness

- 10,039 utterances.
- 10 speakers.
- 360 / 360 fold-level rows.
- 24 / 24 summary rows.
- 18 / 18 paired-delta rows.
- No missing CCC/MAE values.

## Target structure

ICC-like between-speaker fractions:
- Valence: 0.0049.
- Arousal: 0.0447.
- Dominance: 0.0514.

The preregistered >=0.10 between-speaker criterion is not met.

## Raw VAD

Relative minus Absolute:
- Arousal: +0.0773 CCC, 95% CI [0.0568, 0.0967].
- Dominance: +0.0103 [0.0024, 0.0186].
- Valence: +0.0001.

Relative+Baseline minus Relative:
- Arousal: +0.0114, CI crosses zero.
- Dominance: +0.0155, CI [0.0066, 0.0243].
- Valence: +0.0002.

The preregistered raw-baseline utility criterion requiring >=+0.02 for both Arousal and Dominance
is not supported.

## Within-speaker residual VAD

Relative minus Absolute:
- Arousal: +0.1113 CCC, 95% CI [0.1021, 0.1203].
- Dominance: +0.0336 [0.0298, 0.0372].
- Valence: +0.0014.

Relative+Baseline minus Relative:
- Arousal: +0.00059.
- Dominance: +0.00047.
- Valence: +0.00002.

The preregistered within-speaker reference-matching criterion is supported.

## Interpretation

IEMOCAP differs from MSP in raw target structure. Its VAD labels contain very little
between-speaker variance, and raw Arousal already favors Relative prosody. After removing the
speaker target mean, the Relative advantage strengthens further and baseline restoration becomes
negligible.

This is consistent with, rather than contradictory to, the target-reference matching account:
baseline utility depends on whether the target itself contains stable between-speaker structure.

## Verdict

- Validity: valid.
- Decision: mixed.
- Raw baseline replication: not supported.
- Within-speaker Relative replication: supported.
