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

COLS=[
    "sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable",
    "f0_median_hz","pitch_relative_st","pitch_reference_scope"
]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def enroll_indices(df,k,seed):
    out=[]
    for sp,idx in df.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        rng=np.random.default_rng(stable_seed(seed,sp,k,"natural-enrollment"))
        out.extend(rng.choice(idx,size=k,replace=False).tolist())
    return np.array(sorted(out),dtype=int)

def speaker_weights(test):
    counts=test.speaker_id.astype(str).value_counts()
    return test.speaker_id.astype(str).map(lambda s:1.0/counts[s]).to_numpy(float)

def fit_eval(train,test,feature,labels,cfg):
    model=Pipeline([
      ("scale",StandardScaler()),
      ("clf",LogisticRegression(C=float(cfg["C"]),class_weight=cfg["class_weight"],
                                max_iter=int(cfg["max_iter"]),solver="lbfgs"))
    ])
    ytr=train.emotion_5class_candidate.to_numpy()
    yte=test.emotion_5class_candidate.to_numpy()
    model.fit(train[[feature]].to_numpy(np.float32),ytr)
    pred=model.predict(test[[feature]].to_numpy(np.float32))
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

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(src,columns=COLS)
    rows=[]; inv=[]; audit=[]
    nfold=int(cfg["parameters"]["outer_folds"]); K=int(cfg["parameters"]["K"])

    for dataset in cfg["dataset"]["datasets"]:
        d=df[df.dataset.eq(dataset)].copy()
        d=d[d.emotion_training_usable.fillna(False)&d.emotion_5class_candidate.fillna("").ne("")].copy()
        d=d.dropna(subset=["sample_id","speaker_id","f0_median_hz","pitch_relative_st","pitch_reference_scope"])
        d=d[(d.f0_median_hz>0)&d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])].copy()
        d=d[d.speaker_id.astype(str).ne("Unknown")].copy()
        sizes=d.groupby("speaker_id").size()
        keep=sizes[sizes>K].index
        d=d[d.speaker_id.isin(keep)].sort_values("sample_id").reset_index(drop=True)
        d["abs_st"]=12*np.log2(d.f0_median_hz.astype(float))
        d["oracle_rel"]=d.pitch_relative_st.astype(float)
        labels=sorted(d.emotion_5class_candidate.astype(str).unique())
        if len(labels)!=5: raise RuntimeError(f"{dataset}: expected five classes, got {labels}")
        inv.append({
          "dataset":dataset,"rows":len(d),"speakers":d.speaker_id.nunique(),
          "unknown_speaker_rows":int((d.speaker_id.astype(str)=="Unknown").sum()),
          "min_rows_per_speaker":int(d.groupby("speaker_id").size().min()),
          "reference_scopes":"|".join(sorted(d.pitch_reference_scope.unique())),
          **{f"class_{k}":int(v) for k,v in d.emotion_5class_candidate.value_counts().to_dict().items()}
        })

        speakers=sorted(d.speaker_id.astype(str).unique())
        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed))
            sfold=d.speaker_id.astype(str).map(fmap).to_numpy()
            enroll=enroll_indices(d,K,stable_seed(dataset,seed))
            em=np.zeros(len(d),bool); em[enroll]=True
            est={sp:float(g.abs_st.median()) for sp,g in d.loc[em].groupby("speaker_id")}
            for fold in range(nfold):
                tr=(sfold!=fold)&(~em); te=(sfold==fold)&(~em)
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                train_s=set(train.speaker_id.astype(str)); test_s=set(test.speaker_id.astype(str))
                if train_s&test_s: raise RuntimeError("speaker overlap")
                if set(train.emotion_5class_candidate.astype(str).unique())!=set(labels): raise RuntimeError("train class coverage")
                if set(test.emotion_5class_candidate.astype(str).unique())!=set(labels): raise RuntimeError("test class coverage")
                for part in [train,test]:
                    part["kref"]=part.speaker_id.map(est)
                    part["krel"]=part.abs_st-part.kref

                audit.append({
                  "dataset":dataset,"seed":int(seed),"fold":fold,
                  "train_speakers":len(train_s),"test_speakers":len(test_s),
                  "train_rows":len(train),"test_rows":len(test),
                  "speaker_overlap":len(train_s&test_s),
                  "enrollment_eval_overlap":int(np.any(em&(tr|te)))
                })
                for rep,feature in [("absolute","abs_st"),("oracle_relative","oracle_rel"),("kshot_relative","krel")]:
                    sc=fit_eval(train,test,feature,labels,cfg["parameters"]["classifier"])
                    rows.append({"dataset":dataset,"seed":int(seed),"fold":fold,"representation":rep,
                                 "n_train":len(train),"n_test":len(test),**sc})

    m=pd.DataFrame(rows); inventory=pd.DataFrame(inv); aud=pd.DataFrame(audit)
    summary=m.groupby(["dataset","representation"],as_index=False).agg(
      macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
      speaker_balanced_macro_f1_mean=("speaker_balanced_macro_f1","mean"),
      speaker_balanced_macro_f1_std=("speaker_balanced_macro_f1","std"),
      balanced_accuracy_mean=("balanced_accuracy","mean"),accuracy_mean=("accuracy","mean"),n_eval=("macro_f1","size"))
    p=m.pivot_table(index=["dataset","seed","fold"],columns="representation",values="speaker_balanced_macro_f1")
    out=[]
    for dataset,g in p.groupby(level=0):
        for a,b in [("oracle_relative","absolute"),("kshot_relative","absolute"),("kshot_relative","oracle_relative")]:
            v=(g[a]-g[b]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(dataset,a,b))
            out.append({"dataset":dataset,"metric":"speaker_balanced_macro_f1","comparison":f"{a}-{b}",
                        "delta_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    deltas=pd.DataFrame(out)
    inventory.to_csv(root/"data_inventory.csv",index=False)
    aud.to_csv(root/"audit_rows.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"K":K,
      "seeds":cfg["seed"],"outer_folds":nfold,
      "filters":["speaker-specific neutral/shrunk reference","speaker_id != Unknown","speaker rows > K"]
    },indent=2)+"\n")
    print("INVENTORY"); print(inventory.to_string(index=False))
    print("\nSUMMARY"); print(summary.to_string(index=False))
    print("\nPAIRED DELTAS"); print(deltas.to_string(index=False))

if __name__=="__main__":
    main()
