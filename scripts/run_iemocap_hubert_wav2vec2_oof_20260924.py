#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def speaker_id(x):
    m=re.match(r"(Ses\d\d)[FM]_.+_([FM])\d+\.wav$",str(x))
    if not m: raise ValueError(x)
    return m.group(1)+"_"+m.group(2)

def fold_map(speakers,n,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object); np.random.default_rng(seed).shuffle(s)
    return {sp:i%n for i,sp in enumerate(s)}

def weights(df):
    c=df.speaker_id.astype(str).value_counts(); w=df.speaker_id.astype(str).map(lambda s:1/c[s]).to_numpy(float)
    return w/w.mean()

def load_emb(root,name):
    ids=[]; blocks=[]
    for f in sorted((Path(root)/name).glob("train-*.npz")):
        z=np.load(f); ids.extend(z["sample_id"].astype(str)); blocks.append(z["embedding"].astype(np.float32))
    if not blocks: raise RuntimeError(f"no embeddings {name}")
    arr=np.concatenate(blocks); tab=pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))})
    if tab.sample_id.duplicated().any(): raise RuntimeError(f"duplicate embedding ids {name}")
    return tab,arr

def fit(xtr,xte,ytr,w,alpha):
    sc=StandardScaler(); sc.fit(xtr,sample_weight=w); m=Ridge(alpha=alpha); m.fit(sc.transform(xtr),ytr,sample_weight=w)
    return m.predict(sc.transform(xte))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)
    frames=[pd.read_parquet(f,columns=["file","EmoAct","EmoDom"]) for f in sorted(glob.glob(cfg["dataset_glob"]))]
    d=pd.concat(frames,ignore_index=True).dropna().copy(); d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(speaker_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})
    fmap=fold_map(d.speaker_id.unique(),int(cfg["outer_folds"]),int(cfg["split_seed"])); folds=d.speaker_id.map(fmap).to_numpy()
    audit=[]
    for name in cfg["models"]:
        em,X=load_emb(cfg["embedding_root"],name); q=d.merge(em,on="sample_id",validate="one_to_one")
        if len(q)!=10039: raise RuntimeError(f"coverage {name}={len(q)}")
        Xi=X[q.row.to_numpy()]; f=q.speaker_id.map(fmap).to_numpy()
        for target in cfg["targets"]:
            y=q[target].to_numpy(float); pred=np.empty(len(q),float)
            for fold in range(int(cfg["outer_folds"])):
                tr=f!=fold; te=f==fold
                pred[te]=fit(Xi[tr],Xi[te],y[tr],weights(q.loc[tr]),float(cfg["ridge_alpha"]))
                audit.append({"model":name,"target":target,"fold":fold,"train_rows":int(tr.sum()),"test_rows":int(te.sum()),"train_speakers":int(q.loc[tr,"speaker_id"].nunique()),"test_speakers":int(q.loc[te,"speaker_id"].nunique())})
            out=pd.DataFrame({"sample_id":q.sample_id,"speaker_id":q.speaker_id,"target":target,"y_true":y,"pred":pred,"fold":f})
            out.to_parquet(root/f"oof_{name}_{target}.parquet",index=False)
    pd.DataFrame(audit).to_csv(root/"training_audit.csv",index=False)
    meta={"experiment_id":cfg["experiment_id"],"rows":len(d),"speakers":int(d.speaker_id.nunique()),"split_seed":cfg["split_seed"],"ridge_alpha":cfg["ridge_alpha"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    (root/"training_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__": main()

