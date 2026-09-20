# EXPERIMENTS.md

Human-readable derived index only. If this file conflicts with experiments/<ID>/experiment.yaml, the YAML wins.

| ID | Title | Execution | Validity | Decision | Protocol | Parent |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-20260920-01 | Absolute vs speaker-relative prosody across affective-state and speaker-trait tasks | completed | invalid | inconclusive | PROTO-PROSODY-REF-FRAME-V1 | — |
| EXP-20260920-02 | Absolute vs speaker-relative prosody — corrected controlled pilot | completed | valid | pass | PROTO-PROSODY-REF-FRAME-V2 | EXP-20260920-01 |
| EXP-20260920-03 | Frozen WavLM-large plus absolute versus speaker-relative prosody | completed | valid | inconclusive | PROTO-PROSODY-REF-WAVLM-V1 | EXP-20260920-02 |
| EXP-20260920-04 | Layer-wise absolute and speaker-relative pitch structure in WavLM-large | completed | valid | mixed | PROTO-WAVLM-LAYERWISE-PITCH-V1 | EXP-20260920-03 |
| EXP-20260920-05 | Label-free K-shot relative pitch for unseen speakers | completed | valid | mixed | PROTO-UNSEEN-SPEAKER-KSHOT-PITCH-V1 | EXP-20260920-02, EXP-20260920-04 |
| EXP-20260920-06 | What speaker reference does label-free K-shot pitch estimate? | completed | valid | mixed | PROTO-REFERENCE-TARGET-DIAGNOSTIC-V1 | EXP-20260920-05 |
| EXP-20260920-07 | Natural-corpus validation of speaker-relative pitch | completed | valid | inconclusive | PROTO-NATURAL-CORPUS-REFERENCE-V1 | EXP-20260920-05, EXP-20260920-06 |
| EXP-20260920-08 | Why does MSP prefer absolute pitch? | completed | valid | pass | PROTO-PITCH-BASELINE-DECOMP-V1 | EXP-20260920-07 |
| EXP-20260920-09 | Which emotions prefer absolute or relative pitch? | completed | valid | pass | PROTO-PER-EMOTION-CONFLICT-V1 | EXP-20260920-08 |
| EXP-20260920-10 | Do loudness and speaking rate show the same reference-frame heterogeneity as pitch? | completed | valid | mixed | PROTO-MULTI-ATTRIBUTE-EMOTION-CONFLICT-V1 | EXP-20260920-09 |
| EXP-20260920-11 | Can simple statistics predict which prosodic reference frame will work better? | completed | valid | mixed | PROTO-REFERENCE-PREFERENCE-PREDICTOR-V1 | EXP-20260920-09, EXP-20260920-10 |
| EXP-20260920-12 | Can held-out-corpus statistics route each emotion to the better reference frame? | completed | valid | mixed | PROTO-REFERENCE-FRAME-ROUTER-V1 | EXP-20260920-11 |
| EXP-20260920-13 | Can nested confidence-aware routing improve corpus robustness? | completed | valid | reject | PROTO-NESTED-CONFIDENCE-ROUTER-V1 | EXP-20260920-12 |
| EXP-20260920-14 | Are reference-frame effects robust to nonlinear classifiers? | completed | valid | inconclusive | PROTO-CLASSIFIER-FAMILY-ROBUSTNESS-V1 | EXP-20260920-10 |
| EXP-20260920-15 | Which reference-frame effects depend on classifier family? | completed | valid | inconclusive | PROTO-PER-EMOTION-CLASSIFIER-INTERACTION-V1 | EXP-20260920-14 |
| EXP-20260920-16 | Absolute versus speaker-relative prosody for continuous MSP VAD | completed | valid | mixed | PROTO-MSP-VAD-REFERENCE-FRAME-V1 | EXP-20260920-10, EXP-20260920-15 |
