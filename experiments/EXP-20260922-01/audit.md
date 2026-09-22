# Experiment Audit — EXP-20260922-01

## Runtime integrity

- Completed with exit code 0 on 2026-09-22.
- Immutable runtime bundle script/config hashes match the registered source hashes.
- Runtime commit: 713786095900bed424933529fa569a9530701405.
- Dataset inventory: 197,014 rows, 1,915 speakers, 6 interpretable prosodic features.
- Features: pitch level, F0 variability, loudness, speaking rate, pause ratio, voiced ratio.
- Speaker is the independent bootstrap unit; 5,000 speaker-cluster bootstrap replicates.

## Registered acceptance checks

Relative-minus-Absolute slopes:
- Linear Ridge, Arousal: -0.501706, 95% CI [-0.520825, -0.481917].
- Linear Ridge, Dominance: -0.404430, 95% CI [-0.423006, -0.385793].
- Quadratic Ridge, Arousal: -0.507068, 95% CI [-0.525798, -0.488119].
- Quadratic Ridge, Dominance: -0.409179, 95% CI [-0.427194, -0.391303].

All four slopes are negative with 95% confidence intervals fully below zero: supported.

Pure within-speaker endpoint (lambda=0), Relative-minus-Absolute:
- Linear Ridge, Arousal: +0.190055, 95% CI [+0.184007, +0.196011].
- Linear Ridge, Dominance: +0.142450, 95% CI [+0.136166, +0.148738].
- Quadratic Ridge, Arousal: +0.182291, 95% CI [+0.176088, +0.188384].
- Quadratic Ridge, Dominance: +0.135858, 95% CI [+0.129478, +0.141998].

All four lambda=0 effects are positive with 95% confidence intervals fully above zero: supported.

Hybrid-minus-Relative at lambda=1:
- Linear Ridge, Arousal: +0.359250, 95% CI [+0.339510, +0.377933].
- Linear Ridge, Dominance: +0.294348, 95% CI [+0.276632, +0.311522].
- Quadratic Ridge, Arousal: +0.375814, 95% CI [+0.357924, +0.394534].
- Quadratic Ridge, Dominance: +0.311320, 95% CI [+0.295122, +0.328413].

Hybrid exceeds Relative in all four target/model cells at lambda=1: supported.

## Interpretation

The target-reference mechanism is robust when all six validated interpretable prosodic attributes are combined. Relative features are strongly favored when the target is purely within-speaker, while restoring speaker-center information in the Hybrid representation recovers substantial performance when between-speaker target structure dominates. The same qualitative pattern holds for both Linear and Quadratic Ridge.

## Verdict

- Validity: valid.
- Decision: pass.
