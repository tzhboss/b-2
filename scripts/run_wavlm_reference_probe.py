#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from wavlm_reference_probe import prepare, make_folds, matrix, fit_eval, bootstrap

def load_embeddings(root: Path):
    ids=[]; embs=[]
    for p in sorted(root.glob("shard-*-of-*.npz")):
        z=np.load(p,allow_pickle=False)
        ids.extend(z["sample_id"].astype(str).tolist())
        embs.append(z["embedding"].astype(np.float32))
    arr=np.concatenate(embs,axis=0)
    if len(ids)!=len(set(ids)): raise RuntimeError("duplicate embedding ids")
    return ids,arr

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["outputs"]["artifact_root"])
    df=prepare(pd.read_parquet(root/"audio_manifest.parquet"))
    ids,emb=load_embeddings(root/"embeddings")
    pos={s:i for i,s in enumerate(ids)}
    df=df[df.sample_id.astype(str).isin(pos)].copy()
    df["embedding_row"]=df.sample_id.astype(str).map(pos)
    rows=[]; inventory=[]
    reps=cfg["parameters"]["representations"]; n_splits=int(cfg["parameters"]["n_splits"])
    c=cfg["parameters"]["classifier"]

    for dataset in cfg["dataset"]["datasets"]:
        base=df[df.dataset==dataset].copy()
        for task,tcfg in cfg["parameters"]["tasks"].items():
            label=tcfg["label"]; t=base.copy()
            if task=="emotion":
                t=t[t.emotion_training_usable.fillna(False)&t[label].fillna("").ne("")].copy()
            else:
                t=t[t.task_eligible.fillna(False)&t[label].isin(["male","female"])].copy()
            for aset,attrs in cfg["parameters"]["attribute_sets"].items():
                need=[]
                from wavlm_reference_probe import ABS_COLUMNS,REL_COLUMNS
                need=[ABS_COLUMNS[a] for a in attrs]+[REL_COLUMNS[a] for a in attrs]
                sub=t.dropna(subset=need+[label,"speaker_id"]).copy()
                if "pitch" in attrs: sub=sub[sub.f0_median_hz>0]
                if "rate" in attrs: sub=sub[(sub.phoneme_articulation_rate>0)&(sub.rate_relative_ratio>0)]
                sub=sub.sort_values("sample_id").reset_index(drop=True)
                e=emb[sub.embedding_row.to_numpy()]
                rowhash=hashlib.sha256("\n".join(sub.sample_id.astype(str)).encode()).hexdigest()
                labels=sorted(sub[label].astype(str).unique().tolist())
                inventory.append({"dataset":dataset,"task":task,"attribute_set":aset,"n_rows":len(sub),
                                  "n_speakers":sub.speaker_id.nunique(),"n_classes":len(labels),
                                  "row_id_sha256":rowhash})
                for seed in cfg["seed"]:
                    folds=make_folds(sub,label,n_splits,int(seed))
                    for fold in range(n_splits):
                        tr=folds!=fold; te=folds==fold
                        if set(sub.loc[tr,label].astype(str).unique())!=set(labels) or set(sub.loc[te,label].astype(str).unique())!=set(labels):
                            raise RuntimeError(f"incomplete classes {dataset}/{task}/{aset}/{seed}/{fold}")
                        for rep in reps:
                            x=matrix(sub,e,attrs,rep)
                            sc=fit_eval(x[tr],sub.loc[tr,label].to_numpy(),x[te],sub.loc[te,label].to_numpy(),
                                        labels,float(c["alpha"]),c["class_weight"],c["solver"])
                            rows.append({"dataset":dataset,"task":task,"attribute_set":aset,"representation":rep,
                                         "seed":int(seed),"fold":fold,"n_rows":len(sub),**sc})

    m=pd.DataFrame(rows); inv=pd.DataFrame(inventory)
    keys=["dataset","task","attribute_set","representation"]
    summary=m.groupby(keys,as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        accuracy_mean=("accuracy","mean"),balanced_accuracy_mean=("balanced_accuracy","mean"),
        n_eval=("macro_f1","size"),n_rows=("n_rows","max"))
    idx=["dataset","task","attribute_set","seed","fold"]
    p=m.pivot_table(index=idx,columns="representation",values="macro_f1")
    comps=[("relative","absolute"),("hybrid","absolute"),("relative","wavlm_only"),
           ("absolute","wavlm_only"),("hybrid","wavlm_only")]
    ds=[]
    for (dataset,task,aset),g in p.groupby(level=[0,1,2]):
        for a,b in comps:
            d=(g[a]-g[b]).dropna().to_numpy()
            lo,hi=bootstrap(d,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(dataset,task,aset,a,b))
            ds.append({"dataset":dataset,"task":task,"attribute_set":aset,"comparison":f"{a}-{b}",
                       "delta_macro_f1_mean":float(d.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(d)})
    deltas=pd.DataFrame(ds)
    inv.to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    metas=[json.loads(p.read_text()) for p in sorted((root/"embeddings").glob("shard-*-meta.json"))]
    runmeta={"experiment_id":cfg["experiment_id"],"embedding_rows":len(ids),"embedding_dim":int(emb.shape[1]),
             "embedding_dtype_loaded":str(emb.dtype),"embedding_shards":len(metas),
             "model_bin_sha256":metas[0]["model_bin_sha256"] if metas else None,
             "seeds":cfg["seed"],"n_splits":n_splits}
    (root/"run_metadata.json").write_text(json.dumps(runmeta,indent=2)+"\n")
    print(summary.to_string(index=False))
    print("\nPaired deltas:\n",deltas.to_string(index=False))

if __name__=="__main__":
    main()
