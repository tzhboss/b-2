#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
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

def speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    return df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)

def fit_predict(train,test,features,cfg):
    model=Pipeline([
      ("scale",StandardScaler()),
      ("clf",LogisticRegression(C=float(cfg["C"]),class_weight=cfg["class_weight"],
                                max_iter=int(cfg["max_iter"]),solver="lbfgs"))
    ])
    model.fit(train[features].to_numpy(np.float32),train.emotion_5class_candidate.to_numpy())
    return model.predict(test[features].to_numpy(np.float32))

def prepare(raw,dataset):
    d=raw[raw.dataset.eq(dataset)].copy()
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

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)
    nfold=int(cfg["parameters"]["outer_folds"])
    reps={
      "absolute":["absolute_st"],
      "relative":["relative_st"],
      "baseline_only":["baseline_st"],
      "relative_plus_baseline":["relative_st","baseline_st"]
    }
    per=[]; conflicts=[]; inv=[]

    for dataset in cfg["dataset"]["datasets"]:
        d=prepare(raw,dataset)
        labels=sorted(d.emotion_5class_candidate.astype(str).unique())
        if len(labels)!=5: raise RuntimeError(f"{dataset}: expected 5 labels")
        inv.append({"dataset":dataset,"rows":len(d),"speakers":d.speaker_id.nunique(),
                    "min_rows_per_speaker":int(d.groupby("speaker_id").size().min())})
        speakers=sorted(d.speaker_id.astype(str).unique())
        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed))
            sfold=d.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                train=d.loc[sfold!=fold].copy(); test=d.loc[sfold==fold].copy()
                train_s=set(train.speaker_id.astype(str)); test_s=set(test.speaker_id.astype(str))
                if train_s&test_s: raise RuntimeError("speaker leakage")
                # Equal-speaker absolute reference from train only.
                train_sp_medians=train.groupby("speaker_id").absolute_st.median().to_numpy(float)
                corpus_ref=float(np.median(train_sp_medians))
                abs_dev=test.absolute_st.to_numpy(float)-corpus_ref
                rel=test.relative_st.to_numpy(float)
                conflict=(np.sign(abs_dev)*np.sign(rel)<0)&(np.abs(abs_dev)>=float(cfg["parameters"]["conflict_min_abs_st"]))&(np.abs(rel)>=float(cfg["parameters"]["conflict_min_rel_st"]))
                sw=speaker_weights(test)
                y=test.emotion_5class_candidate.astype(str).to_numpy()

                preds={}
                for rep,features in reps.items():
                    pred=fit_predict(train,test,features,cfg["parameters"]["classifier"])
                    preds[rep]=pred
                    for lab in labels:
                        yt=(y==lab).astype(int); yp=(pred==lab).astype(int)
                        per.append({
                          "dataset":dataset,"seed":int(seed),"fold":fold,"emotion":lab,"representation":rep,
                          "speaker_balanced_f1":float(f1_score(yt,yp,sample_weight=sw,zero_division=0)),
                          "speaker_balanced_recall":float(recall_score(yt,yp,sample_weight=sw,zero_division=0)),
                          "support":int(yt.sum())
                        })

                conflict_rows=test.loc[conflict]
                cweights=speaker_weights(conflict_rows) if len(conflict_rows) else np.array([])
                cy=y[conflict]
                row={"dataset":dataset,"seed":int(seed),"fold":fold,"n_test":len(test),
                     "n_conflict":int(conflict.sum()),"conflict_fraction":float(conflict.mean()),
                     "train_corpus_abs_reference_st":corpus_ref}
                for rep,pred in preds.items():
                    cp=pred[conflict]
                    row[f"{rep}_conflict_accuracy"]=float(np.average((cp==cy).astype(float),weights=cweights)) if len(cy) else np.nan
                conflicts.append(row)

    per=pd.DataFrame(per); conflict=pd.DataFrame(conflicts); inventory=pd.DataFrame(inv)
    ps=per.groupby(["dataset","emotion","representation"],as_index=False).agg(
      speaker_balanced_f1_mean=("speaker_balanced_f1","mean"),
      speaker_balanced_f1_std=("speaker_balanced_f1","std"),
      speaker_balanced_recall_mean=("speaker_balanced_recall","mean"),
      n_eval=("speaker_balanced_f1","size"))
    pp=per.pivot_table(index=["dataset","emotion","seed","fold"],columns="representation",values="speaker_balanced_f1")
    deltas=[]
    for (dataset,emotion),g in pp.groupby(level=[0,1]):
        for a,b in [("relative","absolute"),("relative_plus_baseline","relative"),("absolute","relative")]:
            v=(g[a]-g[b]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(dataset,emotion,a,b))
            deltas.append({"dataset":dataset,"emotion":emotion,"comparison":f"{a}-{b}",
                           "delta_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    pdlt=pd.DataFrame(deltas)
    cs=conflict.groupby("dataset",as_index=False).agg(
      conflict_fraction_mean=("conflict_fraction","mean"),
      conflict_fraction_std=("conflict_fraction","std"),
      n_conflict_total=("n_conflict","sum"),
      absolute_conflict_accuracy_mean=("absolute_conflict_accuracy","mean"),
      relative_conflict_accuracy_mean=("relative_conflict_accuracy","mean"),
      baseline_only_conflict_accuracy_mean=("baseline_only_conflict_accuracy","mean"),
      relative_plus_baseline_conflict_accuracy_mean=("relative_plus_baseline_conflict_accuracy","mean"))
    cs["absolute_minus_relative_conflict_accuracy"]=cs.absolute_conflict_accuracy_mean-cs.relative_conflict_accuracy_mean

    inventory.to_csv(root/"data_inventory.csv",index=False)
    per.to_csv(root/"per_emotion_by_fold.csv",index=False)
    ps.to_csv(root/"per_emotion_summary.csv",index=False)
    pdlt.to_csv(root/"per_emotion_deltas.csv",index=False)
    conflict.to_csv(root/"conflict_by_fold.csv",index=False)
    cs.to_csv(root/"conflict_summary.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
      "outer_folds":nfold,"conflict_min_abs_st":cfg["parameters"]["conflict_min_abs_st"],
      "conflict_min_rel_st":cfg["parameters"]["conflict_min_rel_st"],
      "conflict_reference":"train-only median of per-speaker median absolute pitch"
    },indent=2)+"\n")
    print("PER-EMOTION DELTAS"); print(pdlt.to_string(index=False))
    print("\nCONFLICT SUMMARY"); print(cs.to_string(index=False))

if __name__=="__main__":
    main()
