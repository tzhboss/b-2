#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, io, json, gc
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, soundfile as sf, torch, yaml
from transformers import AutoFeatureExtractor, AutoModel

def decode(blob):
    y,sr=sf.read(io.BytesIO(blob),dtype="float32",always_2d=False)
    if y.ndim>1: y=y.mean(axis=1)
    if sr!=16000 or not np.isfinite(y).all(): raise RuntimeError(f"bad audio sr={sr}")
    return y

def extract_model(name,model_id,files,out_root,device,batch_size,cache_dir):
    mroot=out_root/name; mroot.mkdir(parents=True,exist_ok=True)
    fe=AutoFeatureExtractor.from_pretrained(model_id,cache_dir=cache_dir,token=False)
    model=AutoModel.from_pretrained(model_id,cache_dir=cache_dir,token=False).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    inv=[]
    for src in files:
        srcp=Path(src); out=mroot/f"{srcp.stem}.npz"; meta=out.with_suffix(".json")
        if out.exists() and meta.exists():
            z=np.load(out); inv.append({"source":src,"output":str(out),"rows":len(z["sample_id"]),"status":"reused"}); print("reuse",out,flush=True); continue
        pf=pq.ParquetFile(src); ids=[]; blocks=[]; done=0
        for batch in pf.iter_batches(batch_size=batch_size,columns=["file","audio"]):
            d=batch.to_pydict(); aud=[decode(x["bytes"]) for x in d["audio"]]
            inp=fe(aud,sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt")
            iv=inp.input_values.to(device); am=inp.attention_mask.to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.float16):
                h=model(input_values=iv,attention_mask=am).last_hidden_state.float()
            if hasattr(model,"_get_feature_vector_attention_mask"):
                fm=model._get_feature_vector_attention_mask(h.shape[1],am).to(h.dtype)
            else:
                fm=torch.ones(h.shape[:2],device=h.device,dtype=h.dtype)
            emb=(h*fm.unsqueeze(-1)).sum(1)/fm.sum(1,keepdim=True).clamp_min(1)
            if not torch.isfinite(emb).all(): raise RuntimeError("non-finite embedding")
            ids.extend(map(str,d["file"])); blocks.append(emb.cpu().numpy().astype(np.float16)); done+=len(aud)
            if done%250<batch_size: print(f"{name} {srcp.name}: {done}/{pf.metadata.num_rows}",flush=True)
        arr=np.concatenate(blocks); np.savez_compressed(out,sample_id=np.asarray(ids),embedding=arr)
        m={"source":str(srcp.resolve()),"model_name":name,"model_id":model_id,"rows":len(ids),"shape":list(arr.shape),"dtype":str(arr.dtype),"hidden_size":int(model.config.hidden_size),"config_commit":getattr(model.config,"_commit_hash",None)}
        meta.write_text(json.dumps(m,indent=2)+"\n"); inv.append({**m,"output":str(out),"status":"encoded"}); print(json.dumps(m),flush=True)
    (mroot/"inventory.json").write_text(json.dumps(inv,indent=2)+"\n")
    del model,fe; gc.collect(); torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cuda:0"); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); files=sorted(glob.glob(cfg["dataset_glob"]))
    if not files: raise RuntimeError("no parquet")
    out=Path(cfg["embedding_root"]); out.mkdir(parents=True,exist_ok=True)
    cache=Path(cfg["hf_cache"]); cache.mkdir(parents=True,exist_ok=True)
    for name,model_id in cfg["models"].items():
        extract_model(name,model_id,files,out,a.device,int(cfg["batch_size"]),str(cache))
if __name__=="__main__": main()

