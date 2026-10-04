#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

ROOT=Path("/data/lc/tzh")
OUT=ROOT/"results/EXP-20261003-07"
FAIL=OUT/"worker_failures"
MARK=OUT/"tmux_pipeline_complete.json"
PY="/data/conda/envs/test/bin/python"
MERGE=ROOT/"scripts/merge_iemocap_wavlm_e2e_within_aux_20261003.py"
CFG=ROOT/"configs/experiments/EXP-20261003-07.yaml"
TASKS=[(s,f"Ses{i:02d}") for s in [20261002,20261003,20261004] for i in range(1,6)]

while True:
    complete=[]
    for seed,sess in TASKS:
        d=OUT/f"seed-{seed}-{sess}"
        if (d/"predictions.parquet").exists() and (d/"audit.json").exists():
            complete.append((seed,sess))
    failures=sorted(FAIL.glob("*.json")) if FAIL.exists() else []
    status={
      "time":time.time(),
      "complete":[list(x) for x in complete],
      "remaining":[list(x) for x in TASKS if x not in complete],
      "failures":[json.loads(p.read_text()) for p in failures]
    }
    (OUT/"tmux_watch_status.json").write_text(json.dumps(status,indent=2)+"\n")
    print(f"complete={len(complete)}/15 failures={len(failures)}",flush=True)
    if failures:
        raise SystemExit("worker failure marker present")
    if len(complete)==len(TASKS):
        break
    time.sleep(60)

subprocess.run([PY,str(MERGE),"--config",str(CFG)],cwd=str(ROOT),check=True)
payload={"time":time.time(),"complete":True,"merged":True,"tasks":15}
MARK.write_text(json.dumps(payload,indent=2)+"\n")
print("EXP-20261003-07 ALL COMPLETE AND MERGED",flush=True)
