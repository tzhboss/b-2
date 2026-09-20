# Experiment Report — EXP-20260920-01

## Registered Hypothesis

Speaker-relative prosody will improve emotion classification relative to absolute prosody,
whereas absolute prosody will improve gender classification; the magnitude of this trade-off
will vary across pitch, loudness, and rate, with pitch expected to show the strongest
trait-versus-state contrast. Hybrid features may recover complementary information and remain
competitive with the better single reference frame.

## Results

Raw artifacts completed, but no files are promoted to formal results because the audit found a fold-level class-coverage/metric inconsistency in RAVDESS-speech emotion evaluation.

## Observation

ESD-English and MEAD-part0 completed without class-coverage violations. RAVDESS-speech emotion fold 4 omitted neutral for each seed, and the macro-F1 implementation did not force a fixed five-class label universe.

## Supported Claim

The first execution is sufficient to identify and reproduce an evaluation-protocol defect, but it is not valid formal evidence for the registered scientific hypothesis.

## Unsupported Stronger Claim

Do not use EXP-20260920-01 alone to claim that relative pitch improves emotion recognition, that absolute pitch preserves gender information better, or that hybrid features are superior.

## Post-experiment Interpretation

The issue is caused by using five within-speaker-label folds when RAVDESS neutral provides only four retained examples per speaker in the relevant subset. A four-fold correction preserves the intended known-speaker protocol while allowing complete label coverage.

## Decision

inconclusive, because the experiment is invalid under audit.

## Next Step

Run EXP-20260920-02 as a protocol-preserving correction: same datasets, tasks, features, classifier, seeds, and hypotheses; change only to four folds plus explicit full-label coverage validation and fixed-label macro-F1.
