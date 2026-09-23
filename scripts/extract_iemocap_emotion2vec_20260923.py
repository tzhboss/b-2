#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, io, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import yaml

sys.path.insert(0, "/data/lc/s-r/src")
from speech_repr_core.external_baseline import Emotion2VecEncoder


def decode_audio(blob):
    y,sr=sf.read(io.BytesIO(blob),dtype="float32",always_2d=False)
    if y.ndim>1: y=y.mean(axis=1)
    if sr!=16000: raise RuntimeError(f"unexpected sample rate {sr}")
    if not np.isfinite(y).all(): raise RuntimeError("non-finite audio")
    return y


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    out_root=Path(cfg["embedding_file"]).parent/"embeddings"
    out_root.mkdir(parents=True,exist_ok=True)
    files=sorted(glob.glob(cfg["dataset_glob"]))
    if not files: raise RuntimeError("no IEMOCAP parquet files")
    enc=Emotion2VecEncoder(model_source=cfg["model_dir"],device=args.device)
    batch_size=int(cfg.get("batch_size",8))
    inventory=[]
    for src in files:
        srcp=Path(src); out=out_root/f"{srcp.stem}.npz"
        if out.exists():
            z=np.load(out)
            inventory.append({"source":src,"output":str(out),"rows":len(z["sample_id"]),"status":"reused"})
            print(f"reuse {out}",flush=True); continue
        pf=pq.ParquetFile(src)
        ids=[]; blocks=[]; done=0
        for batch in pf.iter_batches(batch_size=batch_size,columns=["file","audio"]):
            d=batch.to_pydict()
            aud=[decode_audio(x["bytes"]) for x in d["audio"]]
            emb=enc.encode(aud)
            if emb.shape!=(len(aud),1024): raise RuntimeError(f"bad embedding shape {emb.shape}")
            ids.extend(map(str,d["file"])); blocks.append(emb.astype(np.float32)); done+=len(aud)
            if done%200<batch_size: print(f"{srcp.name}: {done}/{pf.metadata.num_rows}",flush=True)
        arr=np.concatenate(blocks,axis=0)
        np.savez_compressed(out,sample_id=np.asarray(ids),embedding=arr)
        inventory.append({"source":src,"output":str(out),"rows":len(ids),"status":"encoded"})
        print(f"saved {out} {arr.shape}",flush=True)
    meta=Path(cfg["embedding_file"]).parent/"embedding_inventory.json"
    meta.write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps(inventory,indent=2),flush=True)

if __name__=="__main__": main()
