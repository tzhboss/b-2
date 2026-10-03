#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, glob, hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

def stable_seed(*parts):
    h=hashlib.sha256("::".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:4],"little")

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(int(seed)); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(s):
    vc=pd.Series(s).astype(str).value_counts()
    w=pd.Series(s).astype(str).map(lambda x:1.0/vc[x]).to_numpy(np.float32)
    return w/w.mean()

def load_embeddings(pattern,layer):
    ids=[]; blocks=[]; layers_ref=None
    for f in sorted(glob.glob(pattern)):
        z=np.load(f)
        layers=[int(v) for v in z["layers"].tolist()]
        if layers_ref is None: layers_ref=layers
        elif layers!=layers_ref: raise RuntimeError("layer metadata mismatch")
        if layer not in layers: raise RuntimeError(f"layer {layer} not in {layers}")
        li=layers.index(layer)
        ids.extend(z["sample_id"].astype(str).tolist())
        blocks.append(z["embedding"][:,li,:].astype(np.float32))
    if not blocks: raise RuntimeError("no embeddings")
    X=np.concatenate(blocks,axis=0)
    tab=pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))})
    if tab.sample_id.duplicated().any(): raise RuntimeError("duplicate embedding ids")
    return tab,X

class Net(nn.Module):
    def __init__(self,dim,hidden,dropout):
        super().__init__(); h1,h2=hidden
        self.trunk=nn.Sequential(
            nn.Linear(dim,h1),nn.LayerNorm(h1),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(h1,h2),nn.LayerNorm(h2),nn.GELU()
        )
        self.overall=nn.Linear(h2,3)
        self.within=nn.Linear(h2,3)
    def forward(self,x):
        h=self.trunk(x); return self.overall(h),self.within(h)

def wmse(pred,target,weight):
    per=(pred-target).pow(2).mean(1)
    return (per*weight).sum()/weight.sum()

def train(base_state,variant,xtr,ytr,wtr,sw,cfg,seed,device):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=Net(xtr.shape[1],cfg["hidden_dims"],float(cfg["dropout"])).to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    X=torch.from_numpy(xtr).to(device); Y=torch.from_numpy(ytr).to(device)
    W=torch.from_numpy(wtr).to(device); SW=torch.from_numpy(sw).to(device)
    bs=int(cfg["batch_size"]); rng=np.random.default_rng(seed); fin=np.nan
    for epoch in range(int(cfg["epochs"])):
        order=rng.permutation(len(X)); num=0.; den=0
        model.train()
        for st in range(0,len(X),bs):
            ids=order[st:st+bs]; it=torch.from_numpy(ids).to(device)
            o,w=model(X[it]); ww=SW[it]
            loss=wmse(o,Y[it],ww)
            if variant=="C_within_aux":
                loss=loss+float(cfg["aux_weight"])*wmse(w,W[it],ww)
            elif variant!="A_standard": raise ValueError(variant)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            num+=float(loss.detach().cpu())*len(ids); den+=len(ids)
        fin=num/max(den,1)
    return model,float(fin)

@torch.no_grad()
def predict(model,x,ymean,ystd,device):
    model.eval(); X=torch.from_numpy(x).to(device); out=[]
    for st in range(0,len(X),2048):
        o,_=model(X[st:st+2048]); out.append(o.cpu().numpy())
    z=np.concatenate(out,axis=0)
    return z*ystd+ymean

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--fold",type=int,required=True)
    args=ap.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    seed=int(args.seed); fold=int(args.fold); nfold=int(cfg["outer_folds"])
    if seed not in [int(x) for x in cfg["seeds"]] or not (0<=fold<nfold): raise RuntimeError("invalid shard")
    root=Path(cfg["output_root"])/f"seed-{seed}-fold-{fold}"
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing overwrite {root}")
    root.mkdir(parents=True,exist_ok=True)

    d=pd.read_parquet(cfg["manifest"]).copy()
    d=d.dropna(subset=["sample_id","speaker_id","arousal_mean_1_7","valence_mean_1_7","dominance_mean_1_7"])
    d["speaker_id"]=d.speaker_id.astype(str)
    sizes=d.groupby("speaker_id").size()
    d=d[d.speaker_id.isin(sizes[sizes>10].index)].sort_values("sample_id").reset_index(drop=True)
    fmap=speaker_folds(d.speaker_id.unique(),nfold,seed)
    sf=d.speaker_id.map(fmap).to_numpy()
    tr=sf!=fold; te=sf==fold
    trsp=d.loc[tr,"speaker_id"].to_numpy(); tesp=d.loc[te,"speaker_id"].to_numpy()
    if set(trsp)&set(tesp): raise RuntimeError("speaker leakage")

    tab,Xraw=load_embeddings(cfg["wavlm_embeddings"],int(cfg["wavlm_layer"]))
    idx=tab.set_index("sample_id").loc[d.sample_id.astype(str),"row"].to_numpy(int); Xraw=Xraw[idx]
    xmu=Xraw[tr].mean(0,keepdims=True); xsd=Xraw[tr].std(0,keepdims=True); xsd=np.where(xsd<1e-6,1.,xsd)
    xtr=((Xraw[tr]-xmu)/xsd).astype(np.float32); xte=((Xraw[te]-xmu)/xsd).astype(np.float32)

    Yraw=d[["arousal_mean_1_7","valence_mean_1_7","dominance_mean_1_7"]].to_numpy(np.float32)
    sw=equal_speaker_weights(trsp); y0=Yraw[tr]
    ymean=np.average(y0,axis=0,weights=sw).astype(np.float32)
    yvar=np.average((y0-ymean)**2,axis=0,weights=sw).astype(np.float32)
    ystd=np.sqrt(np.maximum(yvar,1e-6)).astype(np.float32); yz=((y0-ymean)/ystd).astype(np.float32)
    btr=np.empty_like(yz)
    for sp in sorted(set(trsp)):
        m=trsp==sp; btr[m]=yz[m].mean(0)
    wtr=yz-btr

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_seed=stable_seed(seed,fold,"init"); torch.manual_seed(init_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(init_seed)
    base=Net(xtr.shape[1],cfg["hidden_dims"],float(cfg["dropout"])).to(device)
    base_state=copy.deepcopy(base.state_dict()); pc=sum(p.numel() for p in base.parameters())
    variants=["A_standard","C_within_aux"]; preds={}; losses={}
    for v in variants:
        ts=stable_seed(seed,fold,"batches")
        model,loss=train(base_state,v,xtr,yz,wtr,sw,cfg,ts,device)
        preds[v]=predict(model,xte,ymean,ystd,device); losses[v]=loss
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("DONE",seed,fold,v,"loss",round(loss,6),flush=True)

    out=pd.DataFrame({
      "sample_id":d.loc[te,"sample_id"].astype(str).to_numpy(),
      "speaker_id":tesp,
      "seed":seed,"fold":fold,
      "y_arousal":Yraw[te,0],"y_valence":Yraw[te,1],"y_dominance":Yraw[te,2],
      "A_arousal":preds["A_standard"][:,0],"A_valence":preds["A_standard"][:,1],"A_dominance":preds["A_standard"][:,2],
      "C_arousal":preds["C_within_aux"][:,0],"C_valence":preds["C_within_aux"][:,1],"C_dominance":preds["C_within_aux"][:,2],
    })
    out.to_parquet(root/"predictions.parquet",index=False)
    meta={"seed":seed,"fold":fold,"train_rows":int(tr.sum()),"test_rows":int(te.sum()),
          "train_speakers":int(len(set(trsp))),"test_speakers":int(len(set(tesp))),
          "speaker_overlap":0,"parameter_count":int(pc),"losses":losses,"wavlm_layer":int(cfg["wavlm_layer"])}
    (root/"audit.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)

if __name__=="__main__": main()
