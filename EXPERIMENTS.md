# EXPERIMENTS.md
| ID | Title | Execution | Validity | Decision | Protocol | Parent |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-20260920-01 | Absolute vs speaker-relative prosody across affective-state and speaker-trait tasks | completed | invalid | inconclusive | PROTO-PROSODY-REF-FRAME-V1 | — |
| EXP-20260920-02 | Absolute vs speaker-relative prosody — corrected controlled pilot | completed | valid | pass | PROTO-PROSODY-REF-FRAME-V2 | EXP-20260920-01 |
| EXP-20260920-03 | Frozen WavLM-large plus absolute versus speaker-relative prosody | completed | valid | inconclusive | PROTO-PROSODY-REF-WAVLM-V1 | EXP-20260920-02 |
| EXP-20260920-04 | Layer-wise absolute and speaker-relative pitch structure in WavLM-large | completed | valid | inconclusive | PROTO-WAVLM-LAYERWISE-PITCH-V1 | EXP-20260920-03 |
| EXP-20260920-05 | Unseen-speaker K-shot pitch reference estimation | completed | valid | pass | PROTO-UNSEEN-SPEAKER-KSHOT-PITCH-V1 | EXP-20260920-02, EXP-20260920-04 |
