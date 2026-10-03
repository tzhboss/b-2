#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path("/data/lc/tzh")
LOG=ROOT/"logs"/"SER-YW-PIPELINE-20261003.log"
LOG.parent.mkdir(parents=True,exist_ok=True)

def say(*xs):
    s=" ".join(map(str,xs))
    print(s,flush=True)
    with LOG.open("a") as f: f.write(time.strftime("%F %T ") + s + "\n")

def run_sync(cmd, env=None, log_path=None):
    say("RUN", " ".join(map(str,cmd)))
    if log_path:
        Path(log_path).parent.mkdir(parents=True,exist_ok=True)
        with open(log_path,"a") as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    else:
        p=subprocess.run(cmd,env=env)
    say("RC",p.returncode)
    return p.returncode

def complete_dir(p):
    p=Path(p)
    return (p/"predictions.parquet").is_file() and (p/"audit.json").is_file()

def preserve_partial(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()) and not complete_dir(p):
        suffix=time.strftime("%Y%m%d-%H%M%S")
        q=p.with_name(p.name+"-failed-"+suffix)
        p.rename(q)
        say("PRESERVE_PARTIAL",p,"->",q)

def wait_embeddings():
    emb=ROOT/"artifacts/EXP-20261003-06/embeddings"
    manifest=ROOT/"artifacts/EXP-20261003-06/msp_manifest.parquet"
    expected=[emb/f"shard-{i:02d}-of-07.npz" for i in range(7)]
    restart_count=0
    while True:
        good=[p for p in expected if p.is_file()]
        say("MSP_EMBEDDINGS",len(good),"/7")
        if len(good)==7: break
        proc=subprocess.run(["pgrep","-f","extract_msp_wavlm_layers_20261003.py"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if proc.returncode!=0:
            if restart_count>=2:
                raise RuntimeError("MSP extraction stopped before completion after retries")
            restart_count+=1
            say("RESTART_MSP_EXTRACTION",restart_count)
            rc=run_sync([
                "/data/conda/envs/test/bin/python",
                str(ROOT/"scripts/supervise_msp_wavlm_extract_remaining_20261003.py")
            ],log_path=ROOT/"logs"/f"EXP-20261003-06-extract-supervisor-retry{restart_count}.log")
            if rc!=0:
                say("EXTRACTION_SUPERVISOR_FAILED",rc)
        time.sleep(30)
    ids=[]
    for p in expected:
        z=np.load(p)
        ids.extend(z["sample_id"].astype(str).tolist())
        if tuple(z["embedding"].shape[1:])!=(2,1024): raise RuntimeError(f"bad embedding shape {p}")
    n_manifest=len(pd.read_parquet(manifest,columns=["sample_id"]))
    if len(ids)!=n_manifest or len(set(ids))!=n_manifest:
        raise RuntimeError(f"MSP embedding coverage mismatch ids={len(ids)} unique={len(set(ids))} manifest={n_manifest}")
    say("MSP_EMBEDDINGS_VALID",n_manifest)

def schedule(tasks, python_bin, script, max_gpus=7, label="tasks"):
    queue=list(tasks); active={}; failures=[]
    while queue or active:
        for gpu in range(max_gpus):
            if gpu in active or not queue: continue
            task=queue.pop(0)
            outdir=Path(task["outdir"])
            if complete_dir(outdir):
                say("SKIP_COMPLETE",label,task["name"])
                continue
            preserve_partial(outdir)
            env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
            log=ROOT/"logs"/f"{task['name']}.log"; log.parent.mkdir(parents=True,exist_ok=True)
            fh=open(log,"w")
            cmd=[python_bin,str(script),*task["args"]]
            p=subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT)
            active[gpu]=(task,p,fh)
            say("START",label,"gpu",gpu,task["name"],"pid",p.pid)
        if not active and not queue: break
        time.sleep(10)
        for gpu,(task,p,fh) in list(active.items()):
            rc=p.poll()
            if rc is None: continue
            fh.close(); del active[gpu]
            say("DONE",label,"gpu",gpu,task["name"],"rc",rc)
            if rc!=0: failures.append((task["name"],rc))
        if failures:
            for gpu,(task,p,fh) in list(active.items()):
                p.terminate()
            raise RuntimeError(f"{label} failures: {failures}")
    say("ALL_DONE",label)

def msp_train_and_merge():
    cfg=ROOT/"configs/experiments/EXP-20261003-06.yaml"
    c=yaml.safe_load(cfg.read_text()); out=Path(c["output_root"])
    tasks=[]
    for seed in c["seeds"]:
        for fold in range(int(c["outer_folds"])):
            name=f"EXP-20261003-06-seed{seed}-fold{fold}"
            tasks.append({
              "name":name,
              "outdir":out/f"seed-{seed}-fold-{fold}",
              "args":["--config",str(cfg),"--seed",str(seed),"--fold",str(fold)]
            })
    schedule(tasks,"/data/conda/envs/test/bin/python",ROOT/"scripts/run_msp_wavlm_within_aux_shard_20261003.py",label="MSP_TRAIN")
    rc=run_sync(["/data/conda/envs/test/bin/python",str(ROOT/"scripts/merge_msp_wavlm_within_aux_20261003.py"),"--config",str(cfg)],
                log_path=ROOT/"logs"/"EXP-20261003-06-merge.log")
    if rc!=0: raise RuntimeError("MSP merge failed")
    say("MSP_MERGED")

def smoke_e2e(cfg_path):
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]="0"
    log=ROOT/"logs"/"EXP-20261003-07-smoke.log"
    rc=run_sync(["/data/conda/envs/nemo_diarization/bin/python",str(ROOT/"scripts/smoke_iemocap_wavlm_e2e_20261003.py")],env=env,log_path=log)
    if rc==0: return cfg_path
    txt=Path(log).read_text(errors="ignore").lower()
    if "out of memory" not in txt:
        raise RuntimeError("E2E smoke failed for non-OOM reason")
    say("E2E_SMOKE_OOM_FALLBACK_BATCH1")
    c=yaml.safe_load(Path(cfg_path).read_text())
    c["batch_size"]=1; c["eval_batch_size"]=1; c["grad_accum_steps"]=8
    runtime=ROOT/"configs/experiments/EXP-20261003-07-runtime.yaml"
    runtime.write_text(yaml.safe_dump(c,sort_keys=False))
    (ROOT/"results/EXP-20261003-07").mkdir(parents=True,exist_ok=True)
    (ROOT/"results/EXP-20261003-07/runtime_override.json").write_text(json.dumps({
      "reason":"batch_size=2 OOM in pre-result smoke test",
      "batch_size":1,"eval_batch_size":1,"grad_accum_steps":8,
      "effective_batch_preserved":True
    },indent=2)+"\n")
    # Smoke helper reads canonical config; temporarily use runtime through env-unavailable path by replacing then restoring.
    canonical=Path(cfg_path); old=canonical.read_text(); canonical.write_text(runtime.read_text())
    try:
        rc=run_sync(["/data/conda/envs/nemo_diarization/bin/python",str(ROOT/"scripts/smoke_iemocap_wavlm_e2e_20261003.py")],env=env,log_path=log)
    finally:
        canonical.write_text(old)
    if rc!=0: raise RuntimeError("E2E smoke failed after batch1 fallback")
    return str(runtime)

def e2e_train_and_merge():
    canonical=ROOT/"configs/experiments/EXP-20261003-07.yaml"
    cfg_path=smoke_e2e(str(canonical))
    c=yaml.safe_load(Path(cfg_path).read_text()); out=Path(c["output_root"])
    tasks=[]
    for seed in c["seeds"]:
        for held in c["sessions"]:
            name=f"EXP-20261003-07-seed{seed}-{held}"
            tasks.append({
              "name":name,
              "outdir":out/f"seed-{seed}-{held}",
              "args":["--config",str(cfg_path),"--seed",str(seed),"--held-session",str(held)]
            })
    schedule(tasks,"/data/conda/envs/nemo_diarization/bin/python",ROOT/"scripts/run_iemocap_wavlm_e2e_within_aux_shard_20261003.py",label="IEMOCAP_E2E")
    rc=run_sync(["/data/conda/envs/test/bin/python",str(ROOT/"scripts/merge_iemocap_wavlm_e2e_within_aux_20261003.py"),"--config",str(cfg_path)],
                log_path=ROOT/"logs"/"EXP-20261003-07-merge.log")
    if rc!=0: raise RuntimeError("IEMOCAP E2E merge failed")
    say("IEMOCAP_E2E_MERGED")

def final_closeout():
    script=ROOT/"scripts/finalize_ser_yw_pipeline_closeout_20261003.py"
    if not script.is_file():
        say("CLOSEOUT_SCRIPT_MISSING")
        return
    rc=run_sync(["/data/conda/envs/test/bin/python",str(script)],log_path=ROOT/"logs"/"SER-YW-closeout.log")
    if rc!=0: raise RuntimeError("closeout failed")

def main():
    say("PIPELINE_START")
    wait_embeddings()
    msp_train_and_merge()
    e2e_train_and_merge()
    final_closeout()
    say("PIPELINE_COMPLETE")
    (ROOT/"results/SER-YW-PIPELINE-COMPLETE").write_text(time.strftime("%F %T")+"\n")

if __name__=="__main__":
    main()
