#!/usr/bin/env bash
set -euo pipefail
SHARD="${1:?shard index required}"
NUM_SHARDS="${2:-7}"
export CUDA_VISIBLE_DEVICES="${SHARD}"
export WAVLM_MODEL_DIR="/data/lc/models/microsoft-wavlm-large"
exec /data/conda/envs/vllm_cuda13/bin/python \
  /data/lc/tzh/scripts/extract_msp_wavlm_layers_20261003.py \
  --manifest /data/lc/tzh/artifacts/EXP-20261003-06/msp_manifest.parquet \
  --output-dir /data/lc/tzh/artifacts/EXP-20261003-06/embeddings \
  --shard-index "${SHARD}" --num-shards "${NUM_SHARDS}" --batch-size 4 --device cuda:0
