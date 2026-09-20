# CURRENT_STATE.md

## Phase
Unseen-speaker K-shot explicit-pitch experiment completed, audited, and passed. WavLM-conditioned unseen-speaker follow-up is next.

## Valid evidence
- EXP-20260920-02: valid/pass. Prosody-only Absolute/Relative effect.
- EXP-20260920-03: valid/inconclusive. Final-layer WavLM makes explicit Absolute/Relative pitch nearly interchangeable in known-speaker folds.
- EXP-20260920-04: valid/inconclusive primary effect; layer-dependent pitch structure supported.
- EXP-20260920-05: valid/pass. Under speaker-disjoint evaluation, K-shot speaker-relative pitch strongly beats absolute pitch in all three corpora and improves with enrollment budget.

## Main new observation
Relative-K10 minus Absolute macro-F1 is +0.1015 / +0.0922 / +0.0831 on ESD / MEAD / RAVDESS.
Relative-K10 also significantly exceeds Relative-K1 in all three datasets.

## Next legal step
Register a frozen-WavLM unseen-speaker experiment that reuses EXP-05 enrollment pools and target splits exactly.
