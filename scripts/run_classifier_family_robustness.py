#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLS=[
 "sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable",
 "f0_median_hz","pitch_relative_st","pitch_reference_scope",
 "speech_lufs","loudness_relative_lu","loudness_reference_scope",
 "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"
]

ATTRS={
 "pitch": dict(abs="f0_median_hz",rel="pitch_relative_st",scope="pitch_reference_scope",abs_tf="pitch_st",rel_tf="identity"),
 "loudness": dict(abs="speech_lufs",rel="loudness_relative_lu",scope="loudness_reference_scope",abs_tf="identity",rel_tf="identity"),
 "rate": dict(abs="phoneme_articulation_rate",rel="rate_relative_ratio",scope="rate_reference_scope",abs_tf="log",rel_tf="log"),
}

def transform(x,kind):
    x=x.astype(float)
    if kind=="identity": return x
    if kind=="log":
        if not bool((x>0).all()): raise RuntimeError("nonpositive log input")
        return np.log(x)
    if kind=="pitch_st":
        if not bool((x>0).all()): raise RuntimeError("nonpositive pitch")
        return 12*np.log2(x)
    raise ValueError(kind)

def prepare(raw,dataset,attr):
    a=ATTRS[attr]
    d=raw[raw.dataset.eq(dataset)].copy()
    d=d[d.emotion_training_usable.fillna(False)&d.emotion_5class_candidate.fillna("").ne("")].copy()
    d=d.dropna(subset=["sample_id","speaker_id",a["abs"],a["rel"],a["scope"]])
    if dataset in ["msp","meld"]:
        d=d[d[a["scope"]].isin(["speaker_neutral","speaker_neutral_shrunk"])].copy()
        d=d[d.speaker_id.astype(str).ne("Unknown")].copy()
    d["absolute_value"]=transform(d[a["abs"]],a["abs_tf"])
    d["relative_value"]=transform(d[a["rel"]],a["rel_tf"])
    d=d[np.isfinite(d.absolute_value)&np.isfinite(d.relative_value)].copy()
    d["baseline_value"]=d.absolute_value-d.relative_value
    if dataset in ["msp","meld"]:
        sz=d.groupby("speaker_id").size()
        d=d[d.speaker_id.isin(sz[sz>10].index)].copy()
    return d.sort_values("sample_id").reset_index(drop=True)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    return df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)

def fit_predict(train,test,features,family,cfg,seed):
    if family=="logistic":
        model=Pipeline([
          ("scale",StandardScaler()),
          ("clf",LogisticRegression(
            C=float(cfg["C"]),class_weight=cfg["class_weight"],
            max_iter=int(cfg["max_iter"]),solver="lbfgs"))
        ])
        model.fit(train[features].to_numpy(np.float32),train.emotion_5class_candidate.to_numpy())
        return model.predict(test[features].to_numpy(np.float32))
    if family=="hgb":
        model=HistGradientBoostingClassifier(
          learning_rate=float(cfg["learning_rate"]),
          max_iter=int(cfg["max_iter"]),
          max_leaf_nodes=int(cfg["max_leaf_nodes"]),
          min_samples_leaf=int(cfg["min_samples_leaf"]),
          l2_regularization=float(cfg["l2_regularization"]),
          class_weight=cfg["class_weight"],
          early_stopping=bool(cfg["early_stopping"]),
          random_state=int(seed))
        model.fit(train[features].to_numpy(np.float32),train.emotion_5class_candidate.to_numpy())
        return model.predict(test[features].to_numpy(np.float32))
    raise ValueError(family)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)
    nfold=int(cfg["parameters"]["outer_folds"])
    reps={
      "absolute":["absolute_value"],
      "relative":["relative_value"],
      "relative_plus_baseline":["relative_value","baseline_value"]
    }
    rows=[]; inv=[]

    for attr in cfg["parameters"]["attributes"]:
      for dataset in cfg["dataset"]["datasets"]:
        d=prepare(raw,dataset,attr)
        labels=sorted(d.emotion_5class_candidate.astype(str).unique())
        if len(labels)!=5: raise RuntimeError(f"{dataset}/{attr} missing classes")
        row_hash=hashlib.sha256("\n".join(d.sample_id.astype(str)).encode()).hexdigest()
        inv.append({"dataset":dataset,"attribute":attr,"rows":len(d),"speakers":d.speaker_id.nunique(),"row_hash":row_hash})
        speakers=sorted(d.speaker_id.astype(str).unique())
        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed))
            sfold=d.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                train=d.loc[sfold!=fold].copy(); test=d.loc[sfold==fold].copy()
                if set(train.speaker_id.astype(str))&set(test.speaker_id.astype(str)): raise RuntimeError("speaker leakage")
                sw=speaker_weights(test)
                y=test.emotion_5class_candidate.to_numpy()
                for fam,fcfg in cfg["parameters"]["classifiers"].items():
                    for rep,features in reps.items():
                        pred=fit_predict(train,test,features,fam,fcfg,seed)
                        rows.append({
                          "dataset":dataset,"attribute":attr,"seed":int(seed),"fold":fold,
                          "classifier":fam,"representation":rep,
                          "speaker_balanced_macro_f1":float(f1_score(y,pred,labels=labels,average="macro",sample_weight=sw,zero_division=0)),
                          "macro_f1":float(f1_score(y,pred,labels=labels,average="macro",zero_division=0)),
                          "n_train":len(train),"n_test":len(test)
                        })

    m=pd.DataFrame(rows); inv=pd.DataFrame(inv)
    s=m.groupby(["dataset","attribute","classifier","representation"],as_index=False).agg(
      speaker_balanced_macro_f1_mean=("speaker_balanced_macro_f1","mean"),
      speaker_balanced_macro_f1_std=("speaker_balanced_macro_f1","std"),
      macro_f1_mean=("macro_f1","mean"),n_eval=("speaker_balanced_macro_f1","size"))
    p=m.pivot_table(index=["dataset","attribute","classifier","seed","fold"],columns="representation",values="speaker_balanced_macro_f1")
    deltas=[]
    for (ds,attr,fam),g in p.groupby(level=[0,1,2]):
        for a,b in [("relative","absolute"),("relative_plus_baseline","relative")]:
            v=(g[a]-g[b]).to_numpy()
            deltas.append({"dataset":ds,"attribute":attr,"classifier":fam,"comparison":f"{a}-{b}",
                           "delta_mean":float(v.mean()),"delta_std":float(v.std(ddof=1)),"n_pairs":len(v)})
    dlt=pd.DataFrame(deltas)
    ra=dlt[dlt.comparison.eq("relative-absolute")].pivot(index=["dataset","attribute"],columns="classifier",values="delta_mean").reset_index()
    rb=dlt[dlt.comparison.eq("relative_plus_baseline-relative")].pivot(index=["dataset","attribute"],columns="classifier",values="delta_mean").reset_index()
    comp=ra.merge(rb,on=["dataset","attribute"],suffixes=("_ra","_rb"))
    comp["ra_sign_agree"]=(np.sign(comp.logistic_ra)==np.sign(comp.hgb_ra))
    comp["rb_sign_agree"]=(np.sign(comp.logistic_rb)==np.sign(comp.hgb_rb))
    comp["ra_logistic_abs_ge_001"]=comp.logistic_ra.abs()>=.01
    comp["rb_logistic_abs_ge_001"]=comp.logistic_rb.abs()>=.01
    ra_rho=float(spearmanr(comp.logistic_ra,comp.hgb_ra).statistic)
    rb_rho=float(spearmanr(comp.logistic_rb,comp.hgb_rb).statistic)

    inv.to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    s.to_csv(root/"summary.csv",index=False)
    dlt.to_csv(root/"deltas.csv",index=False)
    comp.to_csv(root/"family_comparison.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],
      "source":str(src),"seeds":cfg["seed"],"outer_folds":nfold,
      "relative_absolute_spearman":ra_rho,
      "baseline_addition_spearman":rb_rho,
      "classifier_configs":cfg["parameters"]["classifiers"]
    },indent=2)+"\n")
    print("FAMILY COMPARISON")
    print(comp.to_string(index=False))
    print("\nRA Spearman",ra_rho)
    print("RB Spearman",rb_rho)

if __name__=="__main__":
    main()
