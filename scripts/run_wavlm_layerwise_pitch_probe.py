#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_embeddings(root: Path):
    ids=[]; blocks=[]
    for p in sorted(root.glob("shard-*-of-*.npz")):
        z=np.load(p,allow_pickle=False)
        ids.extend(z["sample_id"].astype(str).tolist())
        blocks.append(z["embedding"].astype(np.float32))
    arr=np.concatenate(blocks,axis=0)
    if len(ids)!=len(set(ids)): raise RuntimeError("duplicate embedding ids")
    return ids,arr

def make_folds(df,label,n_splits,seed):
    rng=np.random.default_rng(seed); folds=np.full(len(df),-1,dtype=int)
    groups=df.reset_index(drop=True).groupby(["speaker_id",label],dropna=False,sort=True).indices
    for idx in groups.values():
        idx=np.asarray(idx); rng.shuffle(idx)
        a=np.arange(len(idx))%n_splits; rng.shuffle(a); folds[idx]=a
    if (folds<0).any(): raise RuntimeError("unassigned fold")
    return folds

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); means=v[idx].mean(1)
    return tuple(map(float,np.quantile(means,[.025,.975])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["outputs"]["artifact_root"])
    manifest=Path(os.environ[cfg["dataset"]["manifest_env"]]).resolve()
    df=pd.read_parquet(manifest).copy()
    df["absolute_pitch_semitone"]=12.0*np.log2(df["f0_median_hz"].astype(float))
    df["relative_pitch_semitone"]=df["pitch_relative_st"].astype(float)
    df["implied_speaker_baseline_semitone"]=df["absolute_pitch_semitone"]-df["relative_pitch_semitone"]

    ids,emb=load_embeddings(root/"layer_embeddings")
    if emb.shape[1:]!=(25,1024): raise RuntimeError(f"unexpected embedding shape {emb.shape}")
    pos={s:i for i,s in enumerate(ids)}
    df=df[df.sample_id.astype(str).isin(pos)].copy()
    df["embedding_row"]=df.sample_id.astype(str).map(pos)

    dec_rows=[]; emo_rows=[]; inv=[]
    n_splits=int(cfg["parameters"]["n_splits"])
    for dataset in cfg["dataset"]["datasets"]:
        base=df[df.dataset==dataset].copy()
        sub=base[
            base.emotion_training_usable.fillna(False)
            & base.emotion_5class_candidate.fillna("").ne("")
        ].dropna(subset=[
            "speaker_id","f0_median_hz","pitch_relative_st",
            "absolute_pitch_semitone","relative_pitch_semitone","implied_speaker_baseline_semitone"
        ]).copy()
        sub=sub[sub.f0_median_hz>0].sort_values("sample_id").reset_index(drop=True)
        labels=sorted(sub.emotion_5class_candidate.astype(str).unique().tolist())
        e=emb[sub.embedding_row.to_numpy()]
        inv.append({
            "dataset":dataset,"n_rows":len(sub),"n_speakers":sub.speaker_id.nunique(),
            "n_emotion_classes":len(labels),
            "row_id_sha256":hashlib.sha256("\n".join(sub.sample_id.astype(str)).encode()).hexdigest()
        })
        targets=["absolute_pitch_semitone","relative_pitch_semitone","implied_speaker_baseline_semitone"]
        Y=sub[targets].to_numpy(dtype=np.float32)

        for seed in cfg["seed"]:
            folds=make_folds(sub,"emotion_5class_candidate",n_splits,int(seed))
            for fold in range(n_splits):
                tr=folds!=fold; te=folds==fold
                if set(sub.loc[tr,"emotion_5class_candidate"].astype(str).unique())!=set(labels): raise RuntimeError("incomplete train labels")
                if set(sub.loc[te,"emotion_5class_candidate"].astype(str).unique())!=set(labels): raise RuntimeError("incomplete test labels")
                for layer in range(25):
                    model=Pipeline([
                        ("scale",StandardScaler()),
                        ("ridge",Ridge(alpha=float(cfg["parameters"]["decodability"]["ridge_alpha"])))
                    ])
                    model.fit(e[tr,layer,:],Y[tr])
                    pred=model.predict(e[te,layer,:])
                    for j,target in enumerate(targets):
                        dec_rows.append({
                            "dataset":dataset,"seed":int(seed),"fold":int(fold),"layer":layer,"target":target,
                            "r2":float(r2_score(Y[te,j],pred[:,j])),
                            "mae":float(mean_absolute_error(Y[te,j],pred[:,j])),"n_rows":len(sub)
                        })
                for layer in cfg["parameters"]["emotion"]["layers"]:
                    for rep in cfg["parameters"]["emotion"]["representations"]:
                        X=e[:,layer,:]
                        if rep=="absolute":
                            X=np.concatenate([X,sub[["absolute_pitch_semitone"]].to_numpy(np.float32)],axis=1)
                        elif rep=="relative":
                            X=np.concatenate([X,sub[["relative_pitch_semitone"]].to_numpy(np.float32)],axis=1)
                        elif rep=="hybrid":
                            X=np.concatenate([X,sub[["absolute_pitch_semitone","relative_pitch_semitone"]].to_numpy(np.float32)],axis=1)
                        elif rep!="wavlm_only":
                            raise ValueError(rep)
                        model=Pipeline([
                            ("scale",StandardScaler()),
                            ("clf",RidgeClassifier(
                                alpha=float(cfg["parameters"]["emotion"]["ridge_alpha"]),
                                class_weight=cfg["parameters"]["emotion"]["class_weight"],
                                solver=cfg["parameters"]["emotion"]["solver"]))
                        ])
                        y=sub.emotion_5class_candidate.to_numpy()
                        model.fit(X[tr],y[tr]); pred=model.predict(X[te])
                        emo_rows.append({
                            "dataset":dataset,"seed":int(seed),"fold":int(fold),"layer":int(layer),"representation":rep,
                            "macro_f1":float(f1_score(y[te],pred,labels=labels,average="macro",zero_division=0)),
                            "accuracy":float(accuracy_score(y[te],pred)),
                            "balanced_accuracy":float(balanced_accuracy_score(y[te],pred)),"n_rows":len(sub)
                        })

    dec=pd.DataFrame(dec_rows); emo=pd.DataFrame(emo_rows); inv=pd.DataFrame(inv)
    decsum=dec.groupby(["dataset","layer","target"],as_index=False).agg(
        r2_mean=("r2","mean"),r2_std=("r2","std"),mae_mean=("mae","mean"),n_eval=("r2","size"))
    emosum=emo.groupby(["dataset","layer","representation"],as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        accuracy_mean=("accuracy","mean"),balanced_accuracy_mean=("balanced_accuracy","mean"),n_eval=("macro_f1","size"))
    pivot=emo.pivot_table(index=["dataset","layer","seed","fold"],columns="representation",values="macro_f1")
    ds=[]
    for (dataset,layer),g in pivot.groupby(level=[0,1]):
        for a,b in [("relative","absolute"),("hybrid","wavlm_only"),("relative","wavlm_only"),("absolute","wavlm_only")]:
            d=(g[a]-g[b]).dropna().to_numpy()
            lo,hi=bootstrap(d,int(cfg["parameters"]["emotion"]["bootstrap_reps"]),stable_seed(dataset,layer,a,b))
            ds.append({"dataset":dataset,"layer":int(layer),"comparison":f"{a}-{b}",
                       "delta_macro_f1_mean":float(d.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(d)})
    deltas=pd.DataFrame(ds)

    inv.to_csv(root/"data_inventory.csv",index=False)
    dec.to_csv(root/"pitch_decodability_by_fold.csv",index=False)
    decsum.to_csv(root/"pitch_decodability_summary.csv",index=False)
    emo.to_csv(root/"emotion_metrics_by_fold.csv",index=False)
    emosum.to_csv(root/"emotion_summary.csv",index=False)
    deltas.to_csv(root/"emotion_paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"embedding_rows":len(ids),"embedding_shape":list(emb.shape),
        "seeds":cfg["seed"],"n_splits":n_splits,"conditional_layers":cfg["parameters"]["emotion"]["layers"]
    },indent=2)+"\n")
    print("DECODABILITY SUMMARY"); print(decsum.to_string(index=False))
    print("\nEMOTION DELTAS"); print(deltas.to_string(index=False))

if __name__=="__main__":
    main()
