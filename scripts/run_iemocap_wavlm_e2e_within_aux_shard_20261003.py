#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, glob, hashlib, json, random, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import yaml
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

def speaker_id(x):
    m=re.match(r"(Ses\d\d)[FM]_.+_([FM])\d+\.wav$",str(x))
    if not m: raise ValueError(x)
    return m.group(1)+"_"+m.group(2)

def session_id(x):
    m=re.match(r"(Ses\d\d)[FM]_",str(x))
    if not m: raise ValueError(x)
    return m.group(1)

def stable_seed(*parts):
    h=hashlib.sha256("::".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:4],"little")

def equal_speaker_weights(s):
    vc=pd.Series(s).astype(str).value_counts()
    w=pd.Series(s).astype(str).map(lambda x:1.0/vc[x]).to_numpy(np.float32)
    return w/w.mean()

def load_audio(path,max_seconds=None):
    y,sr=sf.read(path,dtype="float32",always_2d=False)
    if y.ndim>1: y=y.mean(axis=1)
    if sr!=16000:
        t=torch.from_numpy(np.asarray(y,dtype=np.float32)).unsqueeze(0)
        y=torchaudio.functional.resample(t,sr,16000).squeeze(0).numpy()
    y=np.asarray(y,dtype=np.float32)
    if max_seconds is not None:
        y=y[:int(round(float(max_seconds)*16000))]
    return y

class Model(nn.Module):
    def __init__(self, model_dir, hidden_dims, dropout, gradient_checkpointing=True):
        super().__init__()
        self.wavlm=WavLMModel.from_pretrained(model_dir,local_files_only=True)
        self.wavlm.feature_extractor._freeze_parameters()
        if gradient_checkpointing:
            self.wavlm.gradient_checkpointing_enable()
        h1,h2=hidden_dims
        self.trunk=nn.Sequential(
            nn.Linear(self.wavlm.config.hidden_size,h1),nn.LayerNorm(h1),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(h1,h2),nn.LayerNorm(h2),nn.GELU()
        )
        self.overall=nn.Linear(h2,3)
        self.within=nn.Linear(h2,3)

    def forward(self,input_values,attention_mask):
        out=self.wavlm(input_values=input_values,attention_mask=attention_mask)
        h=out.last_hidden_state
        fm=self.wavlm._get_feature_vector_attention_mask(h.shape[1],attention_mask).to(h.dtype)
        pooled=(h*fm.unsqueeze(-1)).sum(1)/fm.sum(1,keepdim=True).clamp_min(1)
        z=self.trunk(pooled)
        return self.overall(z),self.within(z)

def batch_inputs(fe,paths,device,max_seconds=None):
    audio=[load_audio(p,max_seconds=max_seconds) for p in paths]
    x=fe(audio,sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt")
    return x.input_values.to(device),x.attention_mask.to(device)

def wmse(pred,target,weight):
    per=(pred-target).pow(2).mean(1)
    return (per*weight).sum()/weight.sum()

def make_model(cfg,device,init_seed):
    random.seed(init_seed); np.random.seed(init_seed); torch.manual_seed(init_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(init_seed)
    m=Model(cfg["model_dir"],cfg["hidden_dims"],float(cfg["dropout"]),bool(cfg.get("gradient_checkpointing",True)))
    m.wavlm=m.wavlm.to(device)
    m.trunk=m.trunk.to(device)
    m.overall=m.overall.to(device)
    m.within=m.within.to(device)
    return m

def train_variant(cfg,variant,fe,paths,yz,wtr,sw,orders,device,init_seed):
    model=make_model(cfg,device,init_seed)
    print("MODEL_READY",variant,flush=True)
    enc=[p for n,p in model.named_parameters() if n.startswith("wavlm.") and p.requires_grad]
    head=[p for n,p in model.named_parameters() if not n.startswith("wavlm.") and p.requires_grad]
    opt=torch.optim.AdamW([
        {"params":enc,"lr":float(cfg["encoder_lr"])},
        {"params":head,"lr":float(cfg["head_lr"])}
    ],weight_decay=float(cfg["weight_decay"]))
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
    bs=int(cfg["batch_size"]); accum=int(cfg["grad_accum_steps"]); clip=float(cfg.get("grad_clip",1.0))
    Y=torch.from_numpy(yz); W=torch.from_numpy(wtr); SW=torch.from_numpy(sw)
    final_loss=np.nan
    for epoch,order in enumerate(orders):
        model.train(); opt.zero_grad(set_to_none=True); num=0.; den=0
        for step,st in enumerate(range(0,len(order),bs)):
            ids_np=order[st:st+bs]
            iv,am=batch_inputs(fe,[paths[i] for i in ids_np],device,cfg.get("max_audio_seconds"))
            yb=Y[ids_np].to(device); wb=W[ids_np].to(device); swb=SW[ids_np].to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type=="cuda")):
                o,w=model(iv,am)
                loss=wmse(o,yb,swb)
                if variant=="C_within_aux":
                    loss=loss+float(cfg["aux_weight"])*wmse(w,wb,swb)
                elif variant!="A_standard":
                    raise ValueError(variant)
                loss_scaled=loss/accum
            scaler.scale(loss_scaled).backward()
            do_step=((step+1)%accum==0) or (st+bs>=len(order))
            if do_step:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            num+=float(loss.detach().cpu())*len(ids_np); den+=len(ids_np)
            if step % 200 == 0:
                print("STEP",variant,"epoch",epoch+1,"step",step,"of",(len(order)+bs-1)//bs,"loss",round(float(loss.detach().cpu()),6),flush=True)
        final_loss=num/max(den,1)
        print("EPOCH",variant,epoch+1,"loss",round(final_loss,6),flush=True)
    return model,float(final_loss)

@torch.no_grad()
def predict(cfg,model,fe,paths,device,ymean,ystd):
    model.eval(); bs=int(cfg.get("eval_batch_size",2)); out=[]
    for st in range(0,len(paths),bs):
        iv,am=batch_inputs(fe,paths[st:st+bs],device,cfg.get("max_audio_seconds"))
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type=="cuda")):
            o,_=model(iv,am)
        out.append(o.float().cpu().numpy())
    z=np.concatenate(out,axis=0)
    return z*ystd+ymean

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--held-session",required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    seed=int(args.seed); held=str(args.held_session)
    if seed not in [int(x) for x in cfg["seeds"]]: raise RuntimeError("seed not registered")
    if held not in cfg["sessions"]: raise RuntimeError("held session not registered")
    out=Path(cfg["output_root"])/f"seed-{seed}-{held}"
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refusing overwrite {out}")
    out.mkdir(parents=True,exist_ok=True)

    cols=["file","EmoAct","EmoVal","EmoDom"]
    frames=[pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(cfg["dataset_glob"]))]
    d=pd.concat(frames,ignore_index=True).dropna().copy()
    d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(speaker_id); d["session_id"]=d.file.map(session_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoVal":"valence","EmoDom":"dominance"})
    d["audio_path"]=d.file.map(lambda x:str(Path(cfg["audio_root"])/str(x)))
    if not d.audio_path.map(lambda p:Path(p).is_file()).all(): raise RuntimeError("missing IEMOCAP audio")
    d=d.sort_values("sample_id").reset_index(drop=True)
    tr=d.session_id.to_numpy()!=held; te=~tr
    trsp=d.loc[tr,"speaker_id"].astype(str).to_numpy(); tesp=d.loc[te,"speaker_id"].astype(str).to_numpy()
    if set(trsp)&set(tesp): raise RuntimeError("speaker leakage")

    Yraw=d[["arousal","valence","dominance"]].to_numpy(np.float32)
    sw=equal_speaker_weights(trsp); y0=Yraw[tr]
    ymean=np.average(y0,axis=0,weights=sw).astype(np.float32)
    yvar=np.average((y0-ymean)**2,axis=0,weights=sw).astype(np.float32)
    ystd=np.sqrt(np.maximum(yvar,1e-6)).astype(np.float32); yz=((y0-ymean)/ystd).astype(np.float32)
    btr=np.empty_like(yz)
    for sp in sorted(set(trsp)):
        m=trsp==sp; btr[m]=yz[m].mean(0)
    wtr=yz-btr

    train_paths=d.loc[tr,"audio_path"].astype(str).tolist(); test_paths=d.loc[te,"audio_path"].astype(str).tolist()
    fold=cfg["sessions"].index(held)
    order_rng=np.random.default_rng(stable_seed(seed,fold,"orders"))
    orders=[order_rng.permutation(len(train_paths)) for _ in range(int(cfg["epochs"]))]
    init_seed=stable_seed(seed,fold,"model_init")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fe=Wav2Vec2FeatureExtractor.from_pretrained(cfg["model_dir"],local_files_only=True)

    preds={}; losses={}; trainable=None; total=None
    for variant in ["A_standard","C_within_aux"]:
        model,loss=train_variant(cfg,variant,fe,train_paths,yz,wtr,sw,orders,device,init_seed)
        pred=predict(cfg,model,fe,test_paths,device,ymean,ystd)
        preds[variant]=pred; losses[variant]=loss
        pc=sum(p.numel() for p in model.parameters()); tc=sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=pc if total is None else total; trainable=tc if trainable is None else trainable
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("DONE",seed,held,variant,"loss",round(loss,6),flush=True)

    outdf=pd.DataFrame({
      "sample_id":d.loc[te,"sample_id"].astype(str).to_numpy(),
      "speaker_id":tesp,"session_id":held,"seed":seed,
      "y_arousal":Yraw[te,0],"y_valence":Yraw[te,1],"y_dominance":Yraw[te,2],
      "A_arousal":preds["A_standard"][:,0],"A_valence":preds["A_standard"][:,1],"A_dominance":preds["A_standard"][:,2],
      "C_arousal":preds["C_within_aux"][:,0],"C_valence":preds["C_within_aux"][:,1],"C_dominance":preds["C_within_aux"][:,2],
    })
    outdf.to_parquet(out/"predictions.parquet",index=False)
    audit={
      "seed":seed,"held_session":held,"train_rows":int(tr.sum()),"test_rows":int(te.sum()),
      "train_speakers":len(set(trsp)),"test_speakers":len(set(tesp)),"speaker_overlap":0,
      "model_dir":cfg["model_dir"],"feature_extractor_frozen":True,"transformer_encoder_trainable":True,
      "gradient_checkpointing":bool(cfg.get("gradient_checkpointing",True)),
      "max_audio_seconds":cfg.get("max_audio_seconds"),
      "total_parameters":int(total),"trainable_parameters":int(trainable),"losses":losses,
      "test_speaker_history_or_mean_used_as_input":False
    }
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit),flush=True)

if __name__=="__main__":
    main()