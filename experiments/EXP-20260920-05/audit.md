# Experiment Audit — EXP-20260920-05

## Identity
- Experiment ID: EXP-20260920-05
- Runtime commit: 9907b5c31943883a94138328f21094de3ae99a0b
- Protocol: configs/protocols/unseen_speaker_kshot_pitch_v1.yaml
- Config: configs/experiments/EXP-20260920-05.yaml

## Enrollment Audit
- Total speakers across datasets: 82 = ESD 10 + MEAD 48 + RAVDESS 24.
- Seeds: 3.
- Exactly 10 enrollment utterances per speaker per seed.
- Enrollment rows: 2,460 = 82 × 3 × 10.
- Every enrollment sample exists in the pinned source and has finite F0.
- Independent reconstruction finds zero enrollment/target sample overlap.
- Enrollment selection uses sample identity hashes and speaker ID only; emotion labels are not consulted.

## Evaluation Audit
- Five speaker-disjoint folds per seed.
- ESD test speakers/fold: exactly 2.
- MEAD: 9–10.
- RAVDESS: 4–5.
- Metric rows: 405 = 3 datasets × 3 seeds × 5 folds × 9 representations.
- Metric NaNs: zero.
- Exactly one test-target sample-ID hash per dataset/seed/fold across all 9 representations; no target-set drift across K or reference frame.
- All required emotion classes are present in train/test by execution assertions.

## Baseline Estimation
Mean absolute error versus full-speaker median pitch in semitones:
- ESD: K1 2.895, K2 1.719, K5 1.584, K10 1.317.
- MEAD: K1 1.954, K2 1.282, K5 1.099, K10 0.752.
- RAVDESS: K1 3.953, K2 2.833, K5 2.508, K10 1.214.

The full-speaker median is used only as a diagnostic target and never as a downstream classifier input.

## Registered Criteria
Relative-K10 minus Absolute macro-F1:
- ESD: +0.1015, 95% CI [+0.0692,+0.1337].
- MEAD: +0.0922, 95% CI [+0.0824,+0.1027].
- RAVDESS: +0.0831, 95% CI [+0.0561,+0.1087].

This satisfies the preregistered practical-benefit criterion in all three datasets.

Relative-K10 minus Relative-K1:
- ESD: +0.0816, CI [+0.0364,+0.1228].
- MEAD: +0.0206, CI [+0.0087,+0.0303].
- RAVDESS: +0.0486, CI [+0.0231,+0.0716].

This satisfies the preregistered reference-budget criterion in all three datasets.

## Verdict
- Validity: valid.
- Decision: pass.
- Both preregistered acceptance criteria are met in all three corpora.
