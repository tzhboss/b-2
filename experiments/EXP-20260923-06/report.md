# Experiment Report — EXP-20260923-06

## Result

The frozen-WavLM target-reference intervention was retrained under true IEMOCAP leave-one-session-out
evaluation. Each held-out session contributes both of its speakers only to test; the other four
sessions are used for Ridge training.

All four layer×target cells satisfy the preregistered strongest sign criterion: **5/5 held-out
sessions have negative Relative-minus-Absolute slopes**.

- Layer 12 Arousal: mean session slope -0.07833; session range [-0.17186,-0.02082].
- Layer 12 Dominance: -0.02977; [-0.04900,-0.00763].
- Layer 24 Arousal: -0.08953; [-0.20865,-0.02786].
- Layer 24 Dominance: -0.03585; [-0.06766,-0.00311].

Delete-one-session sensitivity is also sign-stable:

- L12 A: remaining-four mean range [-0.09270,-0.05494].
- L12 D: [-0.03530,-0.02496].
- L24 A: [-0.10495,-0.05975].
- L24 D: [-0.04404,-0.02790].

Thus the WavLM target-reference slope is not explained by conversational partners from the same
IEMOCAP session being split across train and test in the original protocol.

## Boundary

There are only five independent sessions, so this is protocol-robustness evidence rather than a
high-precision population CI. In addition, sklearn emitted ill-conditioned-matrix warnings for the
default Ridge solve on some folds. The direction agrees with EXP-20260921-17's solver-stability
result, but a session-LOSO LSQR sensitivity run is still warranted before using solver-invariant
wording for this exact split protocol.

## Decision

pass; strong session-disjoint sign robustness.
