#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, time
from pathlib import Path
import yaml

ROOT=Path("/data/lc/tzh")
CFG=ROOT/"configs/experiments/EXP-20261003-07.yaml"
PY="/data/conda/envs/nemo_diarization/bin/python"
ENTRY=ROOT/"scripts/run_iemocap_wavlm_e2e_within_aux_shard_20261003.py"
MERGE=ROOT/"scripts/merge_iemocap_wavlm_e2e_within_aux_20261003.py"
LOGDIR=ROOT/"logs"; LOGDIR.mkdir(parents=True,exist_ok=True)
cfg=yaml.safe_load(CFG.read_text())
OUT=Path(cfg["output_root"])
OUT.mkdir(parents=True,exist_ok=True)

tasks=[(int(seed),str(sess)) for seed in cfg["seeds"] for sess in cfg["sessions"]]
pending=[]
for seed,sess in tasks:
    d=OUT/f"seed-{seed}-{sess}"
    if (d/"predictions.parquet").exists() and (d/"audit.json").exists():
        print("SKIP complete",seed,sess,flush=True)
    else:
        if d.exists() and any(d.iterdir()):
            raise RuntimeError(f"partial shard exists and requires audit before retry: {d}")
        pending.append((seed,sess))

running={}
failed=[]

def launch(gpu,task):
    seed,sess=task
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    env["CUDA_DEVICE_MEMORY_SHARED_CACHE"]=f"/tmp/cudevshr-exp07-gpu{gpu}.cache"
    log_path=LOGDIR/f"EXP-20261003-07-{seed}-{sess}.log"
    log=open(log_path,"w")
    cmd=[PY,str(ENTRY),"--config",str(CFG),"--seed",str(seed),"--held-session",sess]
    p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT))
    running[gpu]=(p,task,log,log_path)
    print("START",gpu,seed,sess,p.pid,str(log_path),flush=True)

def wait_initial_model_ready(gpu,timeout=300):
    p,task,log,log_path=running[gpu]
    deadline=time.time()+timeout
    while time.time()<deadline:
        rc=p.poll()
        if rc is not None:
            return False
        try:
            txt=Path(log_path).read_text(errors="ignore")
        except FileNotFoundError:
            txt=""
        if "MODEL_READY A_standard" in txt:
            print("READY",gpu,task[0],task[1],flush=True)
            return True
        time.sleep(2)
    return False

for gpu in range(min(7,len(pending))):
    launch(gpu,pending.pop(0))
    if not wait_initial_model_ready(gpu):
        p,task,log,log_path=running[gpu]
        rc=p.poll()
        failed.append((task[0],task[1],rc if rc is not None else -999,str(log_path)))
        break

while running:
    time.sleep(20)
    for gpu in list(running):
        p,task,log,log_path=running[gpu]
        rc=p.poll()
        if rc is None:
            continue
        log.close()
        seed,sess=task
        print("DONE",gpu,seed,sess,rc,flush=True)
        del running[gpu]
        if rc!=0:
            failed.append((seed,sess,rc,str(log_path)))
        elif pending:
            launch(gpu,pending.pop(0))
            if not wait_initial_model_ready(gpu):
                p2,task2,log2,log_path2=running[gpu]
                rc2=p2.poll()
                failed.append((task2[0],task2[1],rc2 if rc2 is not None else -999,str(log_path2)))
    if failed:
        for gpu,(p,task,log,log_path) in list(running.items()):
            p.terminate()
        for gpu,(p,task,log,log_path) in list(running.items()):
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.kill()
            log.close()
        running.clear()
        break

if failed:
    raise SystemExit(f"failed shards: {failed}")

subprocess.run([PY,str(MERGE),"--config",str(CFG)],cwd=str(ROOT),check=True)
print("ALL_COMPLETE",flush=True)