#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path

root=Path("/data/lc/tzh")
py="/data/conda/envs/nemo_diarization/bin/python"
script=str(root/"scripts/extract_msp_wavlm_layers_20261003.py")
manifest=str(root/"artifacts/EXP-20261003-06/msp_manifest.parquet")
outdir=str(root/"artifacts/EXP-20261003-06/embeddings")
logdir=root/"logs"; logdir.mkdir(parents=True,exist_ok=True)
Path(outdir).mkdir(parents=True,exist_ok=True)
procs=[]
for i in [4,5,6]:
    outfile=Path(outdir)/f"shard-{i:02d}-of-07.npz"
    if outfile.exists():
        print("SKIP existing",outfile,flush=True)
        continue
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(i)
    env["WAVLM_MODEL_DIR"]="/data/lc/models/microsoft-wavlm-large"
    log=open(logdir/f"EXP-20261003-06-extract-{i}.log","w")
    cmd=[py,script,"--manifest",manifest,"--output-dir",outdir,"--shard-index",str(i),"--num-shards","7","--batch-size","4","--device","cuda:0"]
    p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    procs.append((i,p,log))
    print("START",i,p.pid,flush=True)
bad=[]
for i,p,log in procs:
    rc=p.wait(); log.close()
    print("DONE",i,rc,flush=True)
    if rc!=0: bad.append((i,rc))
if bad:
    raise SystemExit(f"failed shards {bad}")
