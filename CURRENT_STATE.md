# CURRENT_STATE.md

## Phase

Clean fixed-cohort K-shot saturation rerun completed and audited.

## Main findings

- EXP-20 is invalid due runtime provenance mismatch and contributes no evidence.
- EXP-20260921-01 is a hash-verified immutable rerun.
- Acoustic-center estimation keeps improving beyond K=10.
- Downstream Arousal/Dominance show much stronger diminishing returns.
- K=10 narrowly misses the preregistered practical plateau for Arousal.
- Descriptively, K=20 is within 0.01 CCC of K=50 for both Arousal and Dominance.

## Next legal step

Use a preregistered speaker-level development/confirmation split: select K=20 as the candidate
plateau only on the development speakers, then independently test K20 versus K50 on untouched
confirmation speakers.
