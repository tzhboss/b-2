# Experiment Audit — EXP-20260920-03

## Identity

- Experiment ID: EXP-20260920-03
- Runtime commit: 68f446a4aae9c18f55dfbee4e78857cbd9e5f1ae
- Branch: main
- Protocol: configs/protocols/prosody_reference_frame_wavlm_v1.yaml
- Experiment config: configs/experiments/EXP-20260920-03.yaml

## Data and Environment

- Audio manifest rows: 50,669.
- ESD-English: 17,500 audio rows.
- MEAD-part0: 31,729 audio rows.
- RAVDESS-speech: 1,440 audio rows.
- Five known MEAD rows without usable audio were excluded and recorded in audio_mapping_exclusions.parquet; those rows also lacked usable prosodic features.
- WavLM checkpoint: /data/lc/models/microsoft-wavlm-large
- WavLM pytorch_model.bin SHA256: fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d
- Extraction environment: torch 2.7.1+cu126; frozen eval-mode WavLM-large.
- Evaluation seeds: 20260920, 20260921, 20260922.
- Evaluation folds: four deterministic within-speaker-label folds.

## Embedding Audit

- Shards: 7.
- Total embedding rows: 50,669.
- Unique sample IDs: 50,669.
- Embedding dimension: 1024.
- Stored dtype: float16; loaded evaluation dtype: float32.
- Missing/extra embedding IDs: zero.
- Non-finite embeddings: zero.
- eval_mode: true for every shard.
- requires_grad_any: false for every shard.
- Checkpoint hash is identical across all shards.

## Evaluation Audit

- Fold-level metric rows: 576 / 576 expected.
- Summary rows: 48 / 48 expected.
- Paired-delta rows: 60 / 60 expected.
- Data-inventory rows: 12 / 12 expected.
- Metric NaNs: zero.
- Duplicate dataset/task/attribute/representation/seed/fold keys: zero.
- Full train/test label coverage failures: zero.
- Same row subset is used for WavLM-only, Absolute, Relative, and Hybrid within each dataset/task/attribute comparison by construction from a single sorted subset and fixed embedding-row mapping.
- StandardScaler and RidgeClassifier are fit within each training fold only.

## Registered Criterion Evaluation

The registered pitch reversal criterion is not satisfied in any dataset.

Relative minus Absolute pitch macro-F1:
- ESD emotion: +0.00047, 95% CI [-0.00038, +0.00129].
- ESD gender: +0.00006, 95% CI [0.00000, +0.00011].
- MEAD emotion: +0.00014, 95% CI [-0.00053, +0.00082].
- MEAD gender: +0.00009, 95% CI [+0.00001, +0.00016].
- RAVDESS emotion: +0.00267, 95% CI [-0.00162, +0.00694].
- RAVDESS gender: +0.00071, 95% CI [-0.00047, +0.00211].

No dataset meets the registered requirement of emotion >= +0.005 with CI above zero and gender <= -0.005 with CI below zero.

The explicit rejection criteria are also not met exactly: not every absolute pitch delta is below 0.0025 because RAVDESS emotion is +0.00267, and there is no statistically resolved same-direction effect across every dataset/task cell.

## Audit Verdict

- Validity: valid
- Decision: inconclusive
- Interpretation boundary: the strong prosody-only Absolute/Relative reversal from EXP-20260920-02 does not replicate as a conditional incremental effect on top of final-layer mean-pooled frozen WavLM-large under this probe.
- Follow-up required: layer-wise WavLM probing, less saturated trait tasks, and unseen-speaker/reference-estimation experiments are needed before concluding that WavLM is fully reference-frame invariant.
