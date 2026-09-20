#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os
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
    "f0_median_hz","pitch_relative_st"
]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers, n_folds, seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed)
    rng.shuffle(s)
    return {sp:i % n_folds for i,sp in enumerate(s)}

def enrollment_indices(df, k, seed):
    chosen=[]
    for sp,idx in df.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        if len(idx)<=k:
            raise RuntimeError(f"speaker {sp} has only {len(idx)} rows for K={k}")
        rng=np.random.default_rng(stable_seed(seed,sp,k,"enrollment"))
        chosen.extend(rng.choice(idx,size=k,replace=False).tolist())
    return np.array(sorted(chosen),dtype=int)

def fit_eval(train_x,train_y,test_x,test_y,labels,cfg):
    model=Pipeline([
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(
            C=float(cfg["C"]),class_weight=cfg["class_weight"],
            max_iter=int(cfg["max_iter"]),solver="lbfgs"))
    ])
    model.fit(train_x,train_y)
    pred=model.predict(test_x)
    return {
        "macro_f1":float(f1_score(test_y,pred,labels=labels,average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(test_y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(test_y,pred))
    }

def bootstrap(v,reps,seed):
    v=np.asarray(v,float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v)))
    m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(src,columns=COLS)
    df=df[df.dataset.isin(cfg["dataset"]["datasets"])].copy()
    df=df[df.emotion_training_usable.fillna(False)&df.emotion_5class_candidate.fillna("").ne("")].copy()
    df=df.dropna(subset=["sample_id","speaker_id","f0_median_hz","pitch_relative_st"])
    df=df[df.f0_median_hz>0].copy()
    df["abs_pitch_st"]=12.0*np.log2(df.f0_median_hz.astype(float))
    df["oracle_baseline_st"]=df.abs_pitch_st-df.pitch_relative_st.astype(float)

    metric_rows=[]; base_rows=[]; audits=[]
    nfold=int(cfg["parameters"]["outer_folds"])
    reps=cfg["parameters"]["representations"]
    clf=cfg["parameters"]["classifier"]

    for dataset in cfg["dataset"]["datasets"]:
        d=df[df.dataset.eq(dataset)].sort_values("sample_id").reset_index(drop=True).copy()
        labels=sorted(d.emotion_5class_candidate.astype(str).unique().tolist())
        speakers=sorted(d.speaker_id.astype(str).unique().tolist())

        # Fixed oracle baseline per speaker: robust median of implied baseline values.
        oracle_by_sp=d.groupby("speaker_id").oracle_baseline_st.median().to_dict()

        for seed in cfg["seed"]:
            foldmap=speaker_folds(speakers,nfold,int(seed))
            speaker_fold=d.speaker_id.astype(str).map(foldmap).to_numpy()
            for fold in range(nfold):
                train_s=set(d.loc[speaker_fold!=fold,"speaker_id"].astype(str))
                test_s=set(d.loc[speaker_fold==fold,"speaker_id"].astype(str))
                if train_s & test_s:
                    raise RuntimeError("speaker leakage")
                for k in cfg["parameters"]["k_values"]:
                    enroll=enrollment_indices(d,int(k),stable_seed(dataset,seed,fold))
                    enroll_mask=np.zeros(len(d),dtype=bool); enroll_mask[enroll]=True

                    # Estimate each speaker baseline only from its K label-free enrollment rows.
                    est={}
                    for sp,g in d.loc[enroll_mask].groupby("speaker_id"):
                        est[sp]=float(np.median(g.abs_pitch_st.to_numpy()))
                    if set(est)!=set(d.speaker_id.unique()):
                        raise RuntimeError("missing speaker enrollment baseline")

                    eval_mask=~enroll_mask
                    train_mask=(speaker_fold!=fold)&eval_mask
                    test_mask=(speaker_fold==fold)&eval_mask
                    train=d.loc[train_mask].copy()
                    test=d.loc[test_mask].copy()

                    # Full class coverage must hold after enrollment removal.
                    if set(train.emotion_5class_candidate.astype(str).unique())!=set(labels):
                        raise RuntimeError(f"incomplete train labels {dataset} {seed} {fold} K{k}")
                    if set(test.emotion_5class_candidate.astype(str).unique())!=set(labels):
                        raise RuntimeError(f"incomplete test labels {dataset} {seed} {fold} K{k}")

                    for part,name in [(train,"train"),(test,"test")]:
                        part["k_baseline_st"]=part.speaker_id.map(est).astype(float)
                        part["k_relative_st"]=part.abs_pitch_st-part.k_baseline_st
                        part["oracle_relative_st"]=part.pitch_relative_st.astype(float)

                    # Baseline errors only for test speakers; one row per held-out speaker.
                    for sp in sorted(test_s):
                        base_rows.append({
                            "dataset":dataset,"seed":int(seed),"fold":fold,"K":int(k),"speaker_id":sp,
                            "estimated_baseline_st":float(est[sp]),
                            "oracle_baseline_st":float(oracle_by_sp[sp]),
                            "abs_error_st":abs(float(est[sp])-float(oracle_by_sp[sp]))
                        })

                    rowhash=hashlib.sha256("\n".join(test.sample_id.astype(str)).encode()).hexdigest()
                    enrollhash=hashlib.sha256("\n".join(d.loc[enroll_mask].sample_id.astype(str)).encode()).hexdigest()
                    audits.append({
                        "dataset":dataset,"seed":int(seed),"fold":fold,"K":int(k),
                        "train_speakers":len(train_s),"test_speakers":len(test_s),
                        "train_rows":len(train),"test_rows":len(test),
                        "test_row_sha256":rowhash,"enrollment_row_sha256":enrollhash,
                        "speaker_overlap":len(train_s&test_s),
                        "enrollment_eval_overlap":int(np.any(enroll_mask & (train_mask|test_mask)))
                    })

                    matrices={
                        "absolute":(["abs_pitch_st"],["abs_pitch_st"]),
                        "oracle_relative":(["oracle_relative_st"],["oracle_relative_st"]),
                        "kshot_relative":(["k_relative_st"],["k_relative_st"]),
                        "oracle_hybrid":(["abs_pitch_st","oracle_relative_st"],["abs_pitch_st","oracle_relative_st"]),
                        "kshot_hybrid":(["abs_pitch_st","k_relative_st"],["abs_pitch_st","k_relative_st"])
                    }
                    ytr=train.emotion_5class_candidate.to_numpy()
                    yte=test.emotion_5class_candidate.to_numpy()
                    for rep in reps:
                        tc,vc=matrices[rep]
                        sc=fit_eval(train[tc].to_numpy(np.float32),ytr,test[vc].to_numpy(np.float32),yte,labels,clf)
                        metric_rows.append({
                            "dataset":dataset,"seed":int(seed),"fold":fold,"K":int(k),
                            "representation":rep,"n_train":len(train),"n_test":len(test),**sc
                        })

    m=pd.DataFrame(metric_rows)
    b=pd.DataFrame(base_rows)
    a=pd.DataFrame(audits)
    summary=m.groupby(["dataset","K","representation"],as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        accuracy_mean=("accuracy","mean"),balanced_accuracy_mean=("balanced_accuracy","mean"),
        n_eval=("macro_f1","size"))
    bsum=b.groupby(["dataset","K"],as_index=False).agg(
        baseline_mae_semitone=("abs_error_st","mean"),
        baseline_median_abs_error_semitone=("abs_error_st","median"),
        baseline_error_std=("abs_error_st","std"),
        n_speaker_evals=("abs_error_st","size"))
    piv=m.pivot_table(index=["dataset","K","seed","fold"],columns="representation",values="macro_f1")
    out=[]
    comps=[("kshot_relative","absolute"),("oracle_relative","absolute"),("kshot_relative","oracle_relative"),
           ("kshot_hybrid","absolute"),("oracle_hybrid","absolute")]
    for (dataset,k),g in piv.groupby(level=[0,1]):
        for x,y in comps:
            v=(g[x]-g[y]).dropna().to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(dataset,k,x,y))
            out.append({"dataset":dataset,"K":int(k),"comparison":f"{x}-{y}",
                        "delta_macro_f1_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    deltas=pd.DataFrame(out)

    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    b.to_csv(root/"baseline_estimation_by_speaker.csv",index=False)
    bsum.to_csv(root/"baseline_summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    a.to_csv(root/"audit_rows.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"source_path":str(src),
        "datasets":cfg["dataset"]["datasets"],"seeds":cfg["seed"],
        "outer_folds":nfold,"k_values":cfg["parameters"]["k_values"],
        "enrollment_selection":"uniform random per speaker without emotion-label conditioning"
    },indent=2)+"\n")

    print("BASELINE SUMMARY")
    print(bsum.to_string(index=False))
    print("\nPAIRED DELTAS")
    print(deltas.to_string(index=False))

if __name__=="__main__":
    main()
