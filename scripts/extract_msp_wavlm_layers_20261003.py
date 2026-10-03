#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

LAYERS=[12,24]

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
    if sr!=16000:
        t=torch.from_numpy(np.asarray(y,dtype=np.float32)).unsqueeze(0)
        y=torchaudio.functional.resample(t,sr,16000).squeeze(0).numpy()
    return np.asarray(y,dtype=np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--shard-index",type=int,required=True)
    ap.add_argument("--num-shards",type=int,required=True)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args()

    model_dir=Path(os.environ.get("WAVLM_MODEL_DIR","/data/lc/models/microsoft-wavlm-large")).resolve()
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(args.manifest)
    df=df.iloc[args.shard_index::args.num_shards].reset_index(drop=True)

    fe=Wav2Vec2FeatureExtractor.from_pretrained(model_dir,local_files_only=True)
    model=WavLMModel.from_pretrained(model_dir,local_files_only=True).to(args.device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    ids=[]; blocks=[]
    for start in range(0,len(df),args.batch_size):
        b=df.iloc[start:start+args.batch_size]
        audio=[load_audio(p) for p in b.audio_path]
        inp=fe(audio,sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt")
        iv=inp.input_values.to(args.device); am=inp.attention_mask.to(args.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.float16):
            out=model(input_values=iv,attention_mask=am,output_hidden_states=True)
        pooled=[]
        for li in LAYERS:
            h=out.hidden_states[li].float()
            fmask=model._get_feature_vector_attention_mask(h.shape[1],am).to(h.dtype)
            p=(h*fmask.unsqueeze(-1)).sum(1)/fmask.sum(1,keepdim=True).clamp_min(1)
            pooled.append(p)
        z=torch.stack(pooled,dim=1)
        if tuple(z.shape[1:])!=(len(LAYERS),1024):
            raise RuntimeError(f"unexpected embedding shape {tuple(z.shape)}")
        if not torch.isfinite(z).all(): raise RuntimeError("non-finite embedding")
        ids.extend(b.sample_id.astype(str).tolist())
        blocks.append(z.cpu().numpy().astype(np.float16))
        if (start//args.batch_size)%100==0:
            print(f"shard={args.shard_index} {min(start+len(b),len(df))}/{len(df)}",flush=True)

    arr=np.concatenate(blocks,axis=0) if blocks else np.empty((0,len(LAYERS),1024),dtype=np.float16)
    np.savez(outdir/f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}.npz",
             sample_id=np.asarray(ids),embedding=arr,layers=np.asarray(LAYERS,dtype=np.int16))
    model_bin=model_dir/"pytorch_model.bin"
    meta={
        "shard_index":args.shard_index,"num_shards":args.num_shards,"rows":len(ids),
        "embedding_shape":list(arr.shape),"embedding_dtype":str(arr.dtype),"layers":LAYERS,
        "model_dir":str(model_dir),
        "model_bin_sha256":sha256(model_bin) if model_bin.exists() else None,
        "torch_version":torch.__version__,"device":args.device,
        "eval_mode":not model.training,"requires_grad_any":any(p.requires_grad for p in model.parameters()),
        "resample_policy":"torchaudio.functional.resample to 16 kHz when needed"
    }
    (outdir/f"shard-{args.shard_index:02d}-meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)

if __name__=="__main__":
    main()
