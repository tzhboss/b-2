#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

def sha256(path: Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_audio(path: str):
    y,sr=sf.read(path,dtype="float32",always_2d=False)
    if y.ndim>1: y=y.mean(axis=1)
    if sr!=16000: raise RuntimeError(f"unexpected sample rate {sr}: {path}")
    return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--shard-index",type=int,required=True)
    ap.add_argument("--num-shards",type=int,required=True)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args()

    model_dir=Path(os.environ["WAVLM_MODEL_DIR"]).resolve()
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(args.manifest)
    df=df.iloc[args.shard_index::args.num_shards].reset_index(drop=True)

    fe=Wav2Vec2FeatureExtractor.from_pretrained(model_dir,local_files_only=True)
    model=WavLMModel.from_pretrained(model_dir,local_files_only=True).to(args.device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    ids=[]; embs=[]
    for start in range(0,len(df),args.batch_size):
        b=df.iloc[start:start+args.batch_size]
        audio=[load_audio(p) for p in b.audio_path]
        inp=fe(audio,sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt")
        input_values=inp.input_values.to(args.device)
        attention_mask=inp.attention_mask.to(args.device)
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
            out=model(input_values=input_values,attention_mask=attention_mask)
        hidden=out.last_hidden_state.float()
        fmask=model._get_feature_vector_attention_mask(hidden.shape[1],attention_mask).to(hidden.dtype)
        pooled=(hidden*fmask.unsqueeze(-1)).sum(1)/fmask.sum(1,keepdim=True).clamp_min(1)
        if not torch.isfinite(pooled).all(): raise RuntimeError("non-finite embedding")
        ids.extend(b.sample_id.astype(str).tolist())
        embs.append(pooled.cpu().numpy().astype(np.float16))
        if (start//args.batch_size)%100==0:
            print(f"shard={args.shard_index} {min(start+len(b),len(df))}/{len(df)}",flush=True)

    arr=np.concatenate(embs,axis=0) if embs else np.empty((0,1024),dtype=np.float16)
    np.savez(outdir/f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}.npz",
             sample_id=np.asarray(ids), embedding=arr)
    meta={
        "shard_index":args.shard_index,"num_shards":args.num_shards,"rows":len(ids),
        "embedding_shape":list(arr.shape),"embedding_dtype":str(arr.dtype),
        "model_dir":str(model_dir),"model_bin_sha256":sha256(model_dir/"pytorch_model.bin"),
        "torch_version":torch.__version__,"device":args.device,"eval_mode":not model.training,
        "requires_grad_any":any(p.requires_grad for p in model.parameters())
    }
    (outdir/f"shard-{args.shard_index:02d}-meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)

if __name__=="__main__":
    main()
