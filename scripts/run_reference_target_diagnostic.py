#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLS=["sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable","f0_median_hz","pitch_relative_st"]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def enrollment_indices(df,k,seed):
    chosen=[]
    for sp,idx in df.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        rng=np.random.default_rng(stable_seed(seed,sp,k,"enrollment"))
        chosen.extend(rng.choice(idx,size=k,replace=False).tolist())
    return np.array(sorted(chosen),dtype=int)

def fit(train_x,train_y,test_x,test_y,labels,cfg):
    model=Pipeline([
      ("scale",StandardScaler()),
      ("clf",LogisticRegression(C=float(cfg["C"]),class_weight=cfg["class_weight"],max_iter=int(cfg["max_iter"]),solver="lbfgs"))
    ])
    model.fit(train_x,train_y)
    pred=model.predict(test_x)
    return float(f1_score(test_y,pred,labels=labels,average="macro",zero_division=0))

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
    df=df[df.dataset.isin(cfg["dataset"]["datasets"])]
    df=df[df.emotion_training_usable.fillna(False)&df.emotion_5class_candidate.fillna("").ne("")].dropna().copy()
    df=df[df.f0_median_hz>0].copy()
    df["abs_st"]=12*np.log2(df.f0_median_hz.astype(float))
    df["neutral_ref_st"]=df.abs_st-df.pitch_relative_st.astype(float)

    errors=[]; metrics=[]
    nfold=int(cfg["parameters"]["outer_folds"])
    for dataset in cfg["dataset"]["datasets"]:
        d=df[df.dataset.eq(dataset)].sort_values("sample_id").reset_index(drop=True).copy()
        labels=sorted(d.emotion_5class_candidate.astype(str).unique())
        neutral=d.groupby("speaker_id").neutral_ref_st.median().to_dict()
        marginal=d.groupby("speaker_id").abs_st.median().to_dict()
        speakers=sorted(d.speaker_id.astype(str).unique())

        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed))
            sfold=d.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                for k in cfg["parameters"]["k_values"]:
                    enroll=enrollment_indices(d,int(k),stable_seed(dataset,seed,fold))
                    em=np.zeros(len(d),bool); em[enroll]=True
                    est={sp:float(g.abs_st.median()) for sp,g in d.loc[em].groupby("speaker_id")}
                    evalm=~em
                    tr=(sfold!=fold)&evalm; te=(sfold==fold)&evalm
                    train=d.loc[tr].copy(); test=d.loc[te].copy()
                    for part in [train,test]:
                        part["kref"]=part.speaker_id.map(est)
                        part["krel"]=part.abs_st-part.kref
                        part["neutral_rel"]=part.abs_st-part.speaker_id.map(neutral)
                        part["marginal_rel"]=part.abs_st-part.speaker_id.map(marginal)

                    test_s=sorted(test.speaker_id.astype(str).unique())
                    for sp in test_s:
                        errors.append({
                            "dataset":dataset,"seed":int(seed),"fold":fold,"K":int(k),"speaker_id":sp,
                            "k_estimate_st":est[sp],"neutral_reference_st":neutral[sp],"marginal_reference_st":marginal[sp],
                            "error_to_neutral_st":abs(est[sp]-neutral[sp]),
                            "error_to_marginal_st":abs(est[sp]-marginal[sp])
                        })

                    ytr=train.emotion_5class_candidate.to_numpy()
                    yte=test.emotion_5class_candidate.to_numpy()
                    for rep,col in [
                        ("absolute","abs_st"),("kshot_relative","krel"),
                        ("neutral_oracle_relative","neutral_rel"),("marginal_oracle_relative","marginal_rel")]:
                        score=fit(train[[col]].to_numpy(np.float32),ytr,test[[col]].to_numpy(np.float32),yte,labels,cfg["parameters"]["classifier"])
                        metrics.append({"dataset":dataset,"seed":int(seed),"fold":fold,"K":int(k),"representation":rep,"macro_f1":score})

    e=pd.DataFrame(errors); m=pd.DataFrame(metrics)
    es=e.groupby(["dataset","K"],as_index=False).agg(
        mae_to_neutral=("error_to_neutral_st","mean"),
        mae_to_marginal=("error_to_marginal_st","mean"),
        median_error_to_neutral=("error_to_neutral_st","median"),
        median_error_to_marginal=("error_to_marginal_st","median"),
        n_speaker_evals=("speaker_id","size"))
    s=m.groupby(["dataset","K","representation"],as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),n_eval=("macro_f1","size"))
    p=m.pivot_table(index=["dataset","K","seed","fold"],columns="representation",values="macro_f1")
    rows=[]
    for (dataset,k),g in p.groupby(level=[0,1]):
        for a,b in [("marginal_oracle_relative","neutral_oracle_relative"),("kshot_relative","marginal_oracle_relative"),("kshot_relative","neutral_oracle_relative")]:
            v=(g[a]-g[b]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(dataset,k,a,b))
            rows.append({"dataset":dataset,"K":int(k),"comparison":f"{a}-{b}","delta_macro_f1_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    dlt=pd.DataFrame(rows)

    e.to_csv(root/"reference_error_by_speaker.csv",index=False)
    es.to_csv(root/"reference_error_summary.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    s.to_csv(root/"summary.csv",index=False)
    dlt.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
        "outer_folds":nfold,"k_values":cfg["parameters"]["k_values"],
        "note":"marginal oracle is diagnostic and uses full eligible speaker pool"
    },indent=2)+"\n")
    print("REFERENCE ERROR SUMMARY"); print(es.to_string(index=False))
    print("\nPAIRED DELTAS"); print(dlt.to_string(index=False))

if __name__=="__main__":
    main()
