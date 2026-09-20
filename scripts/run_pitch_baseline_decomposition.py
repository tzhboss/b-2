#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLS=["sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable",
      "f0_median_hz","pitch_relative_st","pitch_reference_scope"]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def speaker_weights(test):
    c=test.speaker_id.astype(str).value_counts()
    return test.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)

def fit_eval(train,test,features,labels,cfg):
    model=Pipeline([
      ("scale",StandardScaler()),
      ("clf",LogisticRegression(C=float(cfg["C"]),class_weight=cfg["class_weight"],
                                max_iter=int(cfg["max_iter"]),solver="lbfgs"))
    ])
    ytr=train.emotion_5class_candidate.to_numpy()
    yte=test.emotion_5class_candidate.to_numpy()
    model.fit(train[features].to_numpy(np.float32),ytr)
    pred=model.predict(test[features].to_numpy(np.float32))
    sw=speaker_weights(test)
    return {
      "macro_f1":float(f1_score(yte,pred,labels=labels,average="macro",zero_division=0)),
      "speaker_balanced_macro_f1":float(f1_score(yte,pred,labels=labels,average="macro",sample_weight=sw,zero_division=0)),
      "accuracy":float(accuracy_score(yte,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(yte,pred))
    }

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def prepare(df,dataset):
    d=df[df.dataset.eq(dataset)].copy()
    d=d[d.emotion_training_usable.fillna(False)&d.emotion_5class_candidate.fillna("").ne("")].copy()
    d=d.dropna(subset=["sample_id","speaker_id","f0_median_hz","pitch_relative_st"])
    d=d[d.f0_median_hz>0].copy()
    if dataset in ["msp","meld"]:
        d=d[d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])].copy()
        d=d[d.speaker_id.astype(str).ne("Unknown")].copy()
        sizes=d.groupby("speaker_id").size()
        d=d[d.speaker_id.isin(sizes[sizes>10].index)].copy()
    d["absolute_st"]=12*np.log2(d.f0_median_hz.astype(float))
    d["relative_st"]=d.pitch_relative_st.astype(float)
    d["baseline_st"]=d.absolute_st-d.relative_st
    return d.sort_values("sample_id").reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)
    rows=[]; inv=[]; nfold=int(cfg["parameters"]["outer_folds"])

    fmap_features={
      "absolute":["absolute_st"],
      "relative":["relative_st"],
      "baseline_only":["baseline_st"],
      "relative_plus_baseline":["relative_st","baseline_st"]
    }

    for dataset in cfg["dataset"]["datasets"]:
        d=prepare(raw,dataset)
        labels=sorted(d.emotion_5class_candidate.astype(str).unique())
        if len(labels)!=5: raise RuntimeError(f"{dataset} does not have five classes")
        inv.append({"dataset":dataset,"rows":len(d),"speakers":d.speaker_id.nunique(),
                    "min_rows_per_speaker":int(d.groupby("speaker_id").size().min()),
                    "baseline_within_speaker_max_std":float(d.groupby("speaker_id").baseline_st.std().fillna(0).max())})
        speakers=sorted(d.speaker_id.astype(str).unique())
        for seed in cfg["seed"]:
            foldmap=speaker_folds(speakers,nfold,int(seed))
            sfold=d.speaker_id.astype(str).map(foldmap).to_numpy()
            for fold in range(nfold):
                train=d.loc[sfold!=fold].copy(); test=d.loc[sfold==fold].copy()
                if set(train.speaker_id.astype(str))&set(test.speaker_id.astype(str)): raise RuntimeError("speaker leakage")
                if set(train.emotion_5class_candidate.astype(str).unique())!=set(labels): raise RuntimeError("train labels")
                if set(test.emotion_5class_candidate.astype(str).unique())!=set(labels): raise RuntimeError("test labels")
                for rep,features in fmap_features.items():
                    sc=fit_eval(train,test,features,labels,cfg["parameters"]["classifier"])
                    rows.append({"dataset":dataset,"seed":int(seed),"fold":fold,"representation":rep,
                                 "n_train":len(train),"n_test":len(test),**sc})

    m=pd.DataFrame(rows); inventory=pd.DataFrame(inv)
    summary=m.groupby(["dataset","representation"],as_index=False).agg(
      macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
      speaker_balanced_macro_f1_mean=("speaker_balanced_macro_f1","mean"),
      speaker_balanced_macro_f1_std=("speaker_balanced_macro_f1","std"),
      balanced_accuracy_mean=("balanced_accuracy","mean"),accuracy_mean=("accuracy","mean"),n_eval=("macro_f1","size"))
    p=m.pivot_table(index=["dataset","seed","fold"],columns="representation",values="speaker_balanced_macro_f1")
    out=[]
    for dataset,g in p.groupby(level=0):
        for a,b in [("relative_plus_baseline","relative"),("baseline_only","relative"),
                    ("absolute","relative"),("relative_plus_baseline","absolute")]:
            v=(g[a]-g[b]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(dataset,a,b))
            out.append({"dataset":dataset,"comparison":f"{a}-{b}","delta_mean":float(v.mean()),
                        "ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    deltas=pd.DataFrame(out)
    inventory.to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],"outer_folds":nfold
    },indent=2)+"\n")
    print("SUMMARY"); print(summary.to_string(index=False))
    print("\nPAIRED DELTAS"); print(deltas.to_string(index=False))

if __name__=="__main__":
    main()
