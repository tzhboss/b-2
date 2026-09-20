# Experiment Audit — EXP-20260920-07

- Fold-level metric rows: 90 / 90.
- Audit rows: 30 / 30.
- Metric NaNs: zero.
- Train/test speaker overlap: zero.
- Enrollment/evaluation overlap: zero.
- MSP Unknown rows after filtering: zero.
- Retained reference scopes: speaker_neutral and speaker_neutral_shrunk only.
- Minimum retained speaker support before enrollment: 11 rows.

Data:
- MSP: 147,355 rows, 1,784 speakers.
- MELD: 10,579 rows, 59 speakers.

Speaker-balanced Macro-F1 deltas:
- MELD Oracle-relative minus Absolute: +0.0043, 95% CI [-0.0125, +0.0209].
- MSP Oracle-relative minus Absolute: -0.0256, 95% CI [-0.0329, -0.0161].
- MELD K=10 Relative minus Absolute: +0.0046, CI crosses zero.
- MSP K=10 Relative minus Absolute: -0.0756, 95% CI [-0.0811, -0.0703].

Verdict:
- Validity: valid.
- Decision: inconclusive under the preregistered two-corpus generalization criterion.
- Strong supported observation: MSP is a statistically resolved counterexample to a universal relative-pitch advantage.
