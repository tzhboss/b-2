#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, subprocess, time
from pathlib import Path
import yaml

ROOT=Path("/data/lc/tzh")
CFG=ROOT/"configs/experiments/EXP-20261003-07.yaml"
PY="/data/conda/envs/nemo_diarization/bin/python"
ENTRY=ROOT/"scripts/run_iemocap_wavlm_e2e_within_aux_shard_20261003.py"
MERGE=ROOT/"scripts/merge_iemocap_wavlm_e2e_within_aux_20261003.py"
LOGDIR=ROOT/"logs"; LOGDIR.mkdir(parents=True,exist_ok=True)
STATUS=ROOT/"results/EXP-20261003-07/supervisor_status.json"

cfg=yaml.safe_load(CFG.read_text())
OUT=Path(cfg["output_root"]); OUT.mkdir(parents=True,exist_ok=True)
TASKS=[(int(seed),str(sess)) for seed in cfg["seeds"] for sess in cfg["sessions"]]
GPUS=list(range(7))
children={}
retries={task:0 for task in TASKS}

def complete(task):
    seed,sess=task
    d=OUT/f"seed-{seed}-{sess}"
    return (d/"predictions.parquet").exists() and (d/"audit.json").exists()

def partial_nonempty(task):
    seed,sess=task
    d=OUT/f"seed-{seed}-{sess}"
    if not d.exists(): return False
    files=list(d.iterdir())
    return bool(files) and not complete(task)

def scan_external():
    p=subprocess.run(["ps","-eo","pid=,args="],capture_output=True,text=True,check=True)
    out={}
    pat=re.compile(r"^\s*(\d+)\s+.*run_iemocap_wavlm_e2e_within_aux_shard_20261003\.py.*--seed\s+(\d+).*--held-session\s+(Ses\d\d)")
    for line in p.stdout.splitlines():
        m=pat.search(line)
        if not m: continue
        pid=int(m.group(1)); task=(int(m.group(2)),m.group(3))
        try:
            raw=Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            env={}
            for item in raw:
                if b"=" in item:
                    k,v=item.split(b"=",1); env[k.decode(errors="ignore")]=v.decode(errors="ignore")
            gpu=int(env.get("CUDA_VISIBLE_DEVICES","-1").split(",")[0])
        except Exception:
            gpu=-1
        if gpu in GPUS:
            out[task]=(pid,gpu)
    return out

def write_status(external):
    payload={
      "time":time.time(),
      "complete":[list(t) for t in TASKS if complete(t)],
      "external":{f"{t[0]}-{t[1]}":{"pid":pid,"gpu":gpu} for t,(pid,gpu) in external.items()},
      "children":{f"{t[0]}-{t[1]}":{"pid":p.pid,"gpu":gpu,"log":str(log)} for t,(p,gpu,fh,log) in children.items()},
      "pending":[list(t) for t in TASKS if not complete(t) and t not in external and t not in children],
      "retries":{f"{t[0]}-{t[1]}":n for t,n in retries.items() if n},
    }
    STATUS.write_text(json.dumps(payload,indent=2)+"\n")

def launch(task,gpu):
    seed,sess=task
    d=OUT/f"seed-{seed}-{sess}"
    if partial_nonempty(task):
        raise RuntimeError(f"partial nonempty shard requires manual audit: {d}")
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    log=LOGDIR/f"EXP-20261003-07-{seed}-{sess}-supervised.log"
    fh=open(log,"a")
    cmd=[PY,str(ENTRY),"--config",str(CFG),"--seed",str(seed),"--held-session",sess]
    p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=fh,stderr=subprocess.STDOUT)
    children[task]=(p,gpu,fh,log)
    print("LAUNCH",task,"gpu",gpu,"pid",p.pid,flush=True)

while True:
    external=scan_external()

    # Reap our children.
    for task in list(children):
        p,gpu,fh,log=children[task]
        rc=p.poll()
        if rc is None: continue
        fh.close(); del children[task]
        if rc==0 and complete(task):
            print("COMPLETE",task,"gpu",gpu,flush=True)
        else:
            retries[task]+=1
            print("FAILED",task,"gpu",gpu,"rc",rc,"retry",retries[task],"log",log,flush=True)
            if retries[task]>1:
                write_status(external)
                raise SystemExit(f"shard failed twice: {task}; inspect {log}")
            time.sleep(10)

    external=scan_external()
    if all(complete(t) for t in TASKS):
        write_status(external)
        break

    occupied={gpu for _,gpu in external.values()} | {gpu for _,gpu,_,_ in children.values()}
    active_tasks=set(external) | set(children)
    pending=[t for t in TASKS if not complete(t) and t not in active_tasks]
    free=[g for g in GPUS if g not in occupied]

    for gpu,task in zip(free,pending):
        launch(task,gpu)
        time.sleep(8)

    external=scan_external()
    write_status(external)
    time.sleep(20)

subprocess.run([PY,str(MERGE),"--config",str(CFG)],cwd=str(ROOT),check=True)
print("ALL_15_SHARDS_COMPLETE_AND_MERGED",flush=True)
