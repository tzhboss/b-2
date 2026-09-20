#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLS=[
    "sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable",
    "f0_median_hz"
]

def stable_int(*parts):
    h=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],"little")

def assign_speaker_folds(speakers, n_splits, seed):
    sp=np.asarray(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(sp)
    return {s:i % n_splits for i,s in enumerate(sp)}

def choose_enrollment(df, seed, max_k):
    rows=[]
    selected={}
    for speaker,g in df.groupby("speaker_id",sort=True):
        ids=g.sample_id.astype(str).tolist()
        ids=sorted(ids,key=lambda x:stable_int(seed,speaker,x))
        if len(ids)<max_k:
            raise RuntimeError(f"speaker {speaker} has only {len(ids)} F0-valid utterances")
        chosen=ids[:max_k]
        selected[str(speaker)]=chosen
        for rank,sid in enumerate(chosen,1):
            rows.append({"seed":seed,"speaker_id":str(speaker),"rank":rank,"sample_id":sid})
    return selected,pd.DataFrame(rows)

def baseline_map(all_df, enrollment, k):
    abs_lookup=dict(zip(all_df.sample_id.astype(str),all_df.abs_pitch_st.astype(float)))
    out={}
    for sp,ids in enrollment.items():
        vals=[abs_lookup[s] for s in ids[:k]]
        out[sp]=float(np.median(vals))
    return out

def full_speaker_median(all_df):
    return all_df.groupby(all_df.speaker_id.astype(str)).abs_pitch_st.median().to_dict()

def bootstrap(v,reps,seed):
    v=np.asarray(v,float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v)))
    m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

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

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(src,columns=COLS)
    df=df[df.dataset.isin(cfg["dataset"]["datasets"]) & df.f0_median_hz.notna()].copy()
    df["speaker_id"]=df.speaker_id.astype(str)
    df["abs_pitch_st"]=12.0*np.log2(df.f0_median_hz.astype(float))

    enrollment_rows=[]; baseline_rows=[]; metric_rows=[]; target_hash_rows=[]
    K=list(map(int,cfg["parameters"]["k_values"]))
    max_k=int(cfg["parameters"]["enrollment_max_k"])
    n_splits=int(cfg["parameters"]["n_splits"])

    for dataset in cfg["dataset"]["datasets"]:
        all_d=df[df.dataset==dataset].copy().sort_values("sample_id").reset_index(drop=True)
        speakers=sorted(all_d.speaker_id.unique())
        full_med=full_speaker_median(all_d)

        for seed in cfg["seed"]:
            enrollment,enroll_df=choose_enrollment(all_d,int(seed),max_k)
            enroll_df["dataset"]=dataset
            enrollment_rows.append(enroll_df)
            enrollment_ids={sid for ids in enrollment.values() for sid in ids}

            # Fixed targets across all K and representations.
            target=all_d[
                all_d.emotion_training_usable.fillna(False)
                & all_d.emotion_5class_candidate.fillna("").ne("")
                & ~all_d.sample_id.astype(str).isin(enrollment_ids)
            ].copy().sort_values("sample_id").reset_index(drop=True)

            labels=sorted(target.emotion_5class_candidate.astype(str).unique().tolist())
            if len(labels)!=5:
                raise RuntimeError(f"{dataset} expected 5 target emotion classes, got {labels}")

            folds=assign_speaker_folds(speakers,n_splits,int(seed))
            target["fold"]=target.speaker_id.map(folds)

            # Enrollment estimator diagnostics.
            for k in K:
                bm=baseline_map(all_d,enrollment,k)
                for sp in speakers:
                    baseline_rows.append({
                        "dataset":dataset,"seed":int(seed),"speaker_id":sp,"k":k,
                        "estimated_baseline_st":bm[sp],
                        "full_speaker_median_st":float(full_med[sp]),
                        "abs_error_st":abs(bm[sp]-float(full_med[sp]))
                    })

            baseline_by_k={k:baseline_map(all_d,enrollment,k) for k in K}

            for fold in range(n_splits):
                test_speakers={s for s,f in folds.items() if f==fold}
                train_speakers=set(speakers)-test_speakers
                if train_speakers & test_speakers:
                    raise RuntimeError("speaker overlap")
                tr=target[target.speaker_id.isin(train_speakers)].copy()
                te=target[target.speaker_id.isin(test_speakers)].copy()
                if set(tr.emotion_5class_candidate.astype(str).unique())!=set(labels):
                    raise RuntimeError("train class coverage failure")
                if set(te.emotion_5class_candidate.astype(str).unique())!=set(labels):
                    raise RuntimeError("test class coverage failure")

                target_hash=hashlib.sha256("\n".join(te.sample_id.astype(str)).encode()).hexdigest()
                reps=[("absolute",None,None)]
                for k in K:
                    reps.append((f"relative_k{k}",k,"relative"))
                    reps.append((f"hybrid_k{k}",k,"hybrid"))

                for rep,k,kind in reps:
                    if rep=="absolute":
                        Xtr=tr[["abs_pitch_st"]].to_numpy(np.float32)
                        Xte=te[["abs_pitch_st"]].to_numpy(np.float32)
                    else:
                        bm=baseline_by_k[k]
                        tr_rel=tr.abs_pitch_st.to_numpy()-tr.speaker_id.map(bm).to_numpy()
                        te_rel=te.abs_pitch_st.to_numpy()-te.speaker_id.map(bm).to_numpy()
                        if kind=="relative":
                            Xtr=tr_rel[:,None].astype(np.float32)
                            Xte=te_rel[:,None].astype(np.float32)
                        else:
                            Xtr=np.column_stack([tr.abs_pitch_st.to_numpy(),tr_rel]).astype(np.float32)
                            Xte=np.column_stack([te.abs_pitch_st.to_numpy(),te_rel]).astype(np.float32)

                    ytr=tr.emotion_5class_candidate.astype(str).to_numpy()
                    yte=te.emotion_5class_candidate.astype(str).to_numpy()
                    sc=fit_eval(Xtr,ytr,Xte,yte,labels,cfg["parameters"]["classifier"])
                    metric_rows.append({
                        "dataset":dataset,"seed":int(seed),"fold":fold,"representation":rep,
                        "n_train":len(tr),"n_test":len(te),
                        "n_train_speakers":len(train_speakers),"n_test_speakers":len(test_speakers),
                        **sc
                    })
                    target_hash_rows.append({
                        "dataset":dataset,"seed":int(seed),"fold":fold,"representation":rep,
                        "test_sample_id_sha256":target_hash
                    })

    enroll=pd.concat(enrollment_rows,ignore_index=True)
    baselines=pd.DataFrame(baseline_rows)
    metrics=pd.DataFrame(metric_rows)
    hashes=pd.DataFrame(target_hash_rows)

    summary=metrics.groupby(["dataset","representation"],as_index=False).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        accuracy_mean=("accuracy","mean"),balanced_accuracy_mean=("balanced_accuracy","mean"),
        n_eval=("macro_f1","size"))

    pivot=metrics.pivot_table(index=["dataset","seed","fold"],columns="representation",values="macro_f1")
    comps=[]
    for k in K:
        comps += [(f"relative_k{k}","absolute"),(f"hybrid_k{k}","absolute")]
    comps += [("relative_k10","relative_k1"),("hybrid_k10","hybrid_k1")]
    delta_rows=[]
    for dataset,g in pivot.groupby(level=0):
        for a,b in comps:
            d=(g[a]-g[b]).dropna().to_numpy()
            lo,hi=bootstrap(d,int(cfg["parameters"]["bootstrap_reps"]),stable_int(dataset,a,b))
            delta_rows.append({
                "dataset":dataset,"comparison":f"{a}-{b}",
                "delta_macro_f1_mean":float(d.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(d)
            })
    deltas=pd.DataFrame(delta_rows)

    enroll.to_csv(root/"enrollment_inventory.csv",index=False)
    baselines.to_csv(root/"baseline_estimation.csv",index=False)
    metrics.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    hashes.to_csv(root/"target_hash_checks.csv",index=False)
    meta={
        "experiment_id":cfg["experiment_id"],"data_path":str(src),
        "seeds":cfg["seed"],"n_splits":n_splits,"k_values":K,
        "enrollment_max_k":max_k
    }
    (root/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(summary.to_string(index=False))
    print("\nDELTAS\n",deltas.to_string(index=False))
    print("\nBASELINE MAE\n",baselines.groupby(["dataset","k"]).abs_error_st.mean().to_string())

if __name__=="__main__":
    main()
