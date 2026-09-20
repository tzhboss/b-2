#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
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

COLS=["sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable",
      "f0_median_hz","pitch_relative_st","pitch_reference_scope",
      "speech_lufs","loudness_relative_lu","loudness_reference_scope",
      "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"]

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object); rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    return df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)

def prepare(raw,dataset,attr):
    d=raw[raw.dataset.eq(dataset)].copy()
    d=d[d.emotion_training_usable.fillna(False)&d.emotion_5class_candidate.fillna("").ne("")].copy()
    if attr=="pitch":
        a,r,scope="f0_median_hz","pitch_relative_st","pitch_reference_scope"
        d=d.dropna(subset=["sample_id","speaker_id",a,r,scope]); d=d[d[a]>0]
        d["absolute_value"]=12*np.log2(d[a].astype(float)); d["relative_value"]=d[r].astype(float)
    elif attr=="loudness":
        a,r,scope="speech_lufs","loudness_relative_lu","loudness_reference_scope"
        d=d.dropna(subset=["sample_id","speaker_id",a,r,scope])
        d["absolute_value"]=d[a].astype(float); d["relative_value"]=d[r].astype(float)
    else:
        a,r,scope="phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"
        d=d.dropna(subset=["sample_id","speaker_id",a,r,scope]); d=d[(d[a]>0)&(d[r]>0)]
        d["absolute_value"]=np.log(d[a].astype(float)); d["relative_value"]=np.log(d[r].astype(float))
    if dataset in ["msp","meld"]:
        d=d[d[scope].isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d=d[d.speaker_id.astype(str).ne("Unknown")]
        sizes=d.groupby("speaker_id").size()
        d=d[d.speaker_id.isin(sizes[sizes>10].index)]
    d["baseline_value"]=d.absolute_value-d.relative_value
    return d.sort_values("sample_id").reset_index(drop=True)

def fit_predict(train,test,features,family,cfg):
    if family=="logistic":
        model=Pipeline([
            ("scale",StandardScaler()),
            ("clf",LogisticRegression(C=float(cfg["C"]),class_weight=cfg["class_weight"],
                                      max_iter=int(cfg["max_iter"]),solver="lbfgs"))
        ])
    else:
        model=HistGradientBoostingClassifier(
            learning_rate=float(cfg["learning_rate"]),max_iter=int(cfg["max_iter"]),
            max_leaf_nodes=int(cfg["max_leaf_nodes"]),min_samples_leaf=int(cfg["min_samples_leaf"]),
            l2_regularization=float(cfg["l2_regularization"]),class_weight=cfg["class_weight"],
            early_stopping=bool(cfg["early_stopping"]),random_state=0)
    model.fit(train[features].to_numpy(np.float32),train.emotion_5class_candidate.to_numpy())
    return model.predict(test[features].to_numpy(np.float32))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]); root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)
    reps={"absolute":["absolute_value"],"relative":["relative_value"],"relative_plus_baseline":["relative_value","baseline_value"]}
    rows=[]
    for dataset in cfg["dataset"]["datasets"]:
      for attr in cfg["parameters"]["attributes"]:
        d=prepare(raw,dataset,attr); labels=sorted(d.emotion_5class_candidate.astype(str).unique())
        if len(labels)!=5: raise RuntimeError(f"{dataset}/{attr}: labels={labels}")
        speakers=sorted(d.speaker_id.astype(str).unique())
        for seed in cfg["seed"]:
          fmap=speaker_folds(speakers,int(cfg["parameters"]["outer_folds"]),int(seed))
          sf=d.speaker_id.astype(str).map(fmap).to_numpy()
          for fold in range(int(cfg["parameters"]["outer_folds"])):
            tr=d.loc[sf!=fold].copy(); te=d.loc[sf==fold].copy(); y=te.emotion_5class_candidate.astype(str).to_numpy(); sw=speaker_weights(te)
            for family,fcfg in cfg["parameters"]["classifiers"].items():
              for rep,features in reps.items():
                pred=fit_predict(tr,te,features,family,fcfg)
                for lab in labels:
                    yt=(y==lab).astype(int); yp=(pred==lab).astype(int)
                    rows.append({"dataset":dataset,"attribute":attr,"seed":int(seed),"fold":fold,
                                 "classifier":family,"representation":rep,"emotion":lab,
                                 "speaker_balanced_f1":float(f1_score(yt,yp,sample_weight=sw,zero_division=0))})
    per=pd.DataFrame(rows)
    summary=per.groupby(["dataset","attribute","emotion","classifier","representation"],as_index=False).agg(
        f1_mean=("speaker_balanced_f1","mean"),f1_std=("speaker_balanced_f1","std"),n_eval=("speaker_balanced_f1","size"))
    p=per.pivot_table(index=["dataset","attribute","emotion","seed","fold"],columns=["classifier","representation"],values="speaker_balanced_f1")
    out=[]
    for (ds,attr,emo),g in p.groupby(level=[0,1,2]):
        for family in ["logistic","hgb"]:
            ra=(g[(family,"relative")]-g[(family,"absolute")]).to_numpy()
            rb=(g[(family,"relative_plus_baseline")]-g[(family,"relative")]).to_numpy()
            out.append({"dataset":ds,"attribute":attr,"emotion":emo,"classifier":family,
                        "relative_minus_absolute":float(ra.mean()),"baseline_addition":float(rb.mean())})
    deltas=pd.DataFrame(out)
    q=deltas.pivot_table(index=["dataset","attribute","emotion"],columns="classifier",
                         values=["relative_minus_absolute","baseline_addition"]).reset_index()
    q.columns=["_".join([x for x in map(str,c) if x]) if isinstance(c,tuple) else str(c) for c in q.columns]
    q["ra_interaction_hgb_minus_logistic"]=q["relative_minus_absolute_hgb"]-q["relative_minus_absolute_logistic"]
    q["baseline_interaction_hgb_minus_logistic"]=q["baseline_addition_hgb"]-q["baseline_addition_logistic"]
    q["ra_sign_agree"]=np.sign(q["relative_minus_absolute_hgb"])==np.sign(q["relative_minus_absolute_logistic"])
    q["baseline_sign_agree"]=np.sign(q["baseline_addition_hgb"])==np.sign(q["baseline_addition_logistic"])
    q["classifier_sensitive_ra"]=q.ra_interaction_hgb_minus_logistic.abs()>=.05
    q["classifier_sensitive_baseline"]=q.baseline_interaction_hgb_minus_logistic.abs()>=.05

    per.to_csv(root/"per_emotion_by_fold.csv",index=False)
    summary.to_csv(root/"per_emotion_summary.csv",index=False)
    deltas.to_csv(root/"per_emotion_deltas.csv",index=False)
    q.to_csv(root/"classifier_interactions.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
        "outer_folds":cfg["parameters"]["outer_folds"],"classifiers":cfg["parameters"]["classifiers"]
    },indent=2)+"\n")
    print(q.to_string(index=False))
    print("rho_ra",spearmanr(q.relative_minus_absolute_logistic,q.relative_minus_absolute_hgb).statistic)
    print("rho_baseline",spearmanr(q.baseline_addition_logistic,q.baseline_addition_hgb).statistic)

if __name__=="__main__":
    main()
