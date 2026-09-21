#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

def decode_audio(blob):
    y,sr=sf.read(io.BytesIO(blob),dtype="float32",always_2d=False)
    if y.ndim>1:
        y=y.mean(axis=1)
    if sr!=16000:
        raise RuntimeError(f"unexpected sample rate {sr}")
    return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parquet-file",required=True)
    ap.add_argument("--output-file",required=True)
    ap.add_argument("--model-dir",required=True)
    ap.add_argument("--device",required=True)
    ap.add_argument("--batch-size",type=int,default=4)
    args=ap.parse_args()

    layers=[12,24]
    fe=Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir,local_files_only=True)
    model=WavLMModel.from_pretrained(args.model_dir,local_files_only=True).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pf=pq.ParquetFile(args.parquet_file)
    ids=[]; out_blocks=[]; total=0
    for batch in pf.iter_batches(batch_size=args.batch_size,columns=["file","audio"]):
        d=batch.to_pydict()
        audio=[decode_audio(x["bytes"]) for x in d["audio"]]
        inp=fe(audio,sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt")
        iv=inp.input_values.to(args.device)
        am=inp.attention_mask.to(args.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.float16):
            out=model(input_values=iv,attention_mask=am,output_hidden_states=True)
        pooled=[]
        for li in layers:
            h=out.hidden_states[li].float()
            fmask=model._get_feature_vector_attention_mask(h.shape[1],am).to(h.dtype)
            p=(h*fmask.unsqueeze(-1)).sum(1)/fmask.sum(1,keepdim=True).clamp_min(1)
            pooled.append(p)
        z=torch.stack(pooled,dim=1)
        if tuple(z.shape[1:])!=(2,1024):
            raise RuntimeError(f"unexpected embedding shape {tuple(z.shape)}")
        if not torch.isfinite(z).all():
            raise RuntimeError("non-finite WavLM embedding")
        ids.extend(map(str,d["file"]))
        out_blocks.append(z.cpu().numpy().astype(np.float16))
        total+=len(d["file"])
        if total%200<args.batch_size:
            print(f"{Path(args.parquet_file).name}: {total}/{pf.metadata.num_rows}",flush=True)

    arr=np.concatenate(out_blocks,axis=0)
    out=Path(args.output_file); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,sample_id=np.asarray(ids),embedding=arr,layers=np.asarray(layers,dtype=np.int16))
    meta={
      "source":str(Path(args.parquet_file).resolve()),
      "rows":len(ids),
      "shape":list(arr.shape),
      "layers":layers,
      "dtype":str(arr.dtype),
      "device":args.device,
      "model_dir":str(Path(args.model_dir).resolve())
    }
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)

if __name__=="__main__":
    main()
