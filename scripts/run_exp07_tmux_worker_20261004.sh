#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ] || [ $(( ($# - 1) % 2 )) -ne 0 ]; then
  echo "usage: $0 GPU SEED SESSION [SEED SESSION ...]" >&2
  exit 2
fi

GPU="$1"; shift
ROOT=/data/lc/tzh
PY=/data/conda/envs/nemo_diarization/bin/python
ENTRY="$ROOT/scripts/run_iemocap_wavlm_e2e_within_aux_shard_20261003.py"
CFG="$ROOT/configs/experiments/EXP-20261003-07.yaml"
OUTROOT="$ROOT/results/EXP-20261003-07"
LOGROOT="$ROOT/logs"
FAILROOT="$OUTROOT/worker_failures"
mkdir -p "$LOGROOT" "$FAILROOT"
cd "$ROOT"
RUNNER_SHA="$(sha256sum "$ENTRY" | cut -d ' ' -f1)"

run_one() {
  local seed="$1" sess="$2"
  local out="$OUTROOT/seed-${seed}-${sess}"
  local log="$LOGROOT/EXP-20261003-07-${seed}-${sess}-tmux-gpu${GPU}.log"
  if [ -f "$out/predictions.parquet" ] && [ -f "$out/audit.json" ]; then
    echo "SKIP COMPLETE gpu=$GPU seed=$seed session=$sess"
    return 0
  fi
  if [ -d "$out" ] && find "$out" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
    echo "REFUSE PARTIAL NONEMPTY $out" | tee -a "$log"
    printf '{"gpu":%s,"seed":%s,"session":"%s","reason":"partial_nonempty","log":"%s"}\n' "$GPU" "$seed" "$sess" "$log" > "$FAILROOT/${seed}-${sess}.json"
    return 20
  fi

  local attempt rc
  for attempt in 1 2; do
    echo "START gpu=$GPU seed=$seed session=$sess attempt=$attempt $(date -Is)" | tee -a "$log"
    set +e
    SER_SUPERVISED_RUN=1 SER_RUNNER_SHA256="$RUNNER_SHA" CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ENTRY" --config "$CFG" --seed "$seed" --held-session "$sess" >> "$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ] && [ -f "$out/predictions.parquet" ] && [ -f "$out/audit.json" ]; then
      echo "DONE gpu=$GPU seed=$seed session=$sess attempt=$attempt $(date -Is)" | tee -a "$log"
      rm -f "$FAILROOT/${seed}-${sess}.json"
      return 0
    fi
    echo "FAILED gpu=$GPU seed=$seed session=$sess attempt=$attempt rc=$rc $(date -Is)" | tee -a "$log"
    if [ -d "$out" ] && find "$out" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
      break
    fi
    sleep 15
  done
  printf '{"gpu":%s,"seed":%s,"session":"%s","reason":"run_failed","rc":%s,"log":"%s"}\n' "$GPU" "$seed" "$sess" "$rc" "$log" > "$FAILROOT/${seed}-${sess}.json"
  return "$rc"
}

while [ "$#" -gt 0 ]; do
  seed="$1"; sess="$2"; shift 2
  run_one "$seed" "$sess"
done

echo "WORKER COMPLETE gpu=$GPU $(date -Is)"