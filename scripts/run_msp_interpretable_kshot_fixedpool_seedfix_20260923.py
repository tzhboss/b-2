#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

COLS=[
 "sample_id","dataset","speaker_id",
 "valence_mean_1_7","arousal_mean_1_7","dominance_mean_1_7",
 "f0_median_hz","pitch_reference_scope",
 "speech_lufs","loudness_reference_scope",
 "phoneme_articulation_rate","rate_reference_scope",
 "f0_iqr_semitone","pause_ratio","voiced_ratio"
]

def stable_seed(*parts):
    payload="|".join(map(str,parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def enrollment_indices(df,k,seed):
    out=[]
    for sp,idx in df.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        rng=np.random.default_rng(stable_seed(seed,sp,"fixedpool-enrollment"))
        order=rng.permutation(idx)
        out.extend(order[:k].tolist())
    return np.array(sorted(out),dtype=int)

def equal_speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p); dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def fit_predict(train,test,cols,ytr,wtr,alpha):
    sc=StandardScaler()
    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
    sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha)
    model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def bootstrap(v,reps,seed):
    v=np.asarray(v,float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v)))
    means=v[idx].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]])
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)

    d=pd.read_parquet(src,columns=COLS)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["f0var_abs"]=d.f0_iqr_semitone.astype(float)
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d["pause_abs"]=d.pause_ratio.astype(float)
    d["voiced_abs"]=d.voiced_ratio.astype(float)
    feature_names=["pitch","f0var","loud","rate","pause","voiced"]
    abs_cols=["pitch_abs","f0var_abs","loud_abs","rate_abs","pause_abs","voiced_abs"]
    requested=list(cfg["parameters"].get("features",feature_names))
    if requested!=feature_names: raise ValueError(f"unsupported feature set/order: {requested}")

    sizes=d.groupby("speaker_id").size()
    keep=sizes[sizes>int(cfg["parameters"]["min_support_exclusive"])].index
    d=d[d.speaker_id.isin(keep)].sort_values("sample_id").reset_index(drop=True)
    speakers=sorted(d.speaker_id.astype(str).unique())
    marginal={sp:g[abs_cols].median().to_numpy(float) for sp,g in d.groupby("speaker_id")}
    target_cols=list(cfg["parameters"]["targets"].values())
    nfold=int(cfg["parameters"]["outer_folds"])
    reserve_k=int(cfg["parameters"]["reserve_k"])

    metrics=[]; errors=[]; audits=[]; enrollment_audits=[]
    for seed in cfg["seed"]:
        fmap=speaker_folds(speakers,nfold,int(seed))
        sf=d.speaker_id.astype(str).map(fmap).to_numpy()

        seed_key=stable_seed(cfg["experiment_id"],seed)
        reserve=enrollment_indices(d,reserve_k,seed_key)
        reserve_mask=np.zeros(len(d),bool); reserve_mask[reserve]=True

        for k in cfg["parameters"]["k_values"]:
            enroll=enrollment_indices(d,int(k),seed_key)
            em=np.zeros(len(d),bool); em[enroll]=True
            if np.any(em & ~reserve_mask):
                raise RuntimeError("nested K enrollment escaped the fixed 50-utterance reserve")
            selected_ids=sorted(d.loc[em,"sample_id"].astype(str).tolist())
            enrollment_audits.append({
                "seed":int(seed),"K":int(k),"n_rows":len(selected_ids),
                "selection_sha256":hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
                "seed_key":int(seed_key)
            })
            kest={sp:g[abs_cols].median().to_numpy(float) for sp,g in d.loc[em].groupby("speaker_id")}
            oof_rows=[]

            for fold in range(nfold):
                tr=(sf!=fold)&(~reserve_mask); te=(sf==fold)&(~reserve_mask)
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                if np.any(reserve_mask&(tr|te)):
                    raise RuntimeError("reserved enrollment overlap")
                train_s=set(train.speaker_id.astype(str)); test_s=set(test.speaker_id.astype(str))
                if train_s&test_s:
                    raise RuntimeError("speaker leakage")

                for part in [train,test]:
                    kc=np.vstack([kest[sp] for sp in part.speaker_id])
                    mc=np.vstack([marginal[sp] for sp in part.speaker_id])
                    av=part[abs_cols].to_numpy(float)
                    for j,name in enumerate(feature_names):
                        part[f"{name}_k_base"]=kc[:,j]
                        part[f"{name}_k_rel"]=av[:,j]-kc[:,j]
                        part[f"{name}_marg_base"]=mc[:,j]
                        part[f"{name}_marg_rel"]=av[:,j]-mc[:,j]

                for sp in sorted(test_s):
                    for j,name in enumerate(feature_names):
                        errors.append({
                            "seed":int(seed),"fold":fold,"K":int(k),"speaker_id":sp,"attribute":name,
                            "error_to_marginal":abs(float(kest[sp][j])-float(marginal[sp][j]))
                        })

                reps={
                    "absolute":abs_cols,
                    "kshot_hybrid":[f"{n}_k_rel" for n in feature_names]+[f"{n}_k_base" for n in feature_names],
                    "marginal_oracle_hybrid":[f"{n}_marg_rel" for n in feature_names]+[f"{n}_marg_base" for n in feature_names]
                }
                ytr=train[target_cols].to_numpy(float); yte=test[target_cols].to_numpy(float)
                wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                preds={rep:fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"])) for rep,cols in reps.items()}
                for rep,pred in preds.items():
                    for j,tname in enumerate(cfg["parameters"]["targets"]):
                        metrics.append({
                            "seed":int(seed),"fold":fold,"K":int(k),"representation":rep,"target":tname,
                            "ccc":weighted_ccc(yte[:,j],pred[:,j],wte)
                        })

                for pos,row in enumerate(test.itertuples(index=False)):
                    for j,tname in enumerate(cfg["parameters"]["targets"]):
                        oof_rows.append({
                            "sample_id":str(row.sample_id),"speaker_id":str(row.speaker_id),
                            "seed":int(seed),"fold":fold,"K":int(k),"target":tname,
                            "y_true":float(yte[pos,j]),
                            "pred_absolute":float(preds["absolute"][pos,j]),
                            "pred_kshot_hybrid":float(preds["kshot_hybrid"][pos,j]),
                            "pred_marginal_oracle_hybrid":float(preds["marginal_oracle_hybrid"][pos,j])
                        })

                audits.append({
                    "seed":int(seed),"fold":fold,"K":int(k),
                    "train_rows":len(train),"test_rows":len(test),
                    "train_speakers":len(train_s),"test_speakers":len(test_s),
                    "speaker_overlap":len(train_s&test_s),
                    "reserve_eval_overlap":int(np.any(reserve_mask&(tr|te))),
                    "test_row_sha256":hashlib.sha256("\n".join(test.sample_id.astype(str)).encode()).hexdigest()
                })

            pd.DataFrame(oof_rows).to_parquet(root/f"paired_oof_seed_{int(seed)}_K{int(k)}.parquet",index=False)

    m=pd.DataFrame(metrics); e=pd.DataFrame(errors); a=pd.DataFrame(audits)
    es=e.groupby(["K","attribute"],as_index=False).agg(mae_to_marginal=("error_to_marginal","mean"))
    summary=m.groupby(["K","target","representation"],as_index=False).agg(
        ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),n_eval=("ccc","size"))

    # Formal uncertainty is computed from saved paired OOF predictions by the separate speaker-cluster entrypoint.
    delta=pd.DataFrame()
    kd=pd.DataFrame()
    ea=pd.DataFrame(enrollment_audits)
    reserve_hashes=ea[ea.K.eq(reserve_k)].selection_sha256
    if reserve_hashes.nunique()!=len(cfg["seed"]):
        raise RuntimeError("corrected enrollment seeds did not produce distinct K=50 reserves")

    pd.DataFrame([{
        "rows":len(d),"speakers":len(speakers),
        "min_rows_per_speaker":int(d.groupby("speaker_id").size().min()),
        "reserved_rows_per_speaker":reserve_k
    }]).to_csv(root/"data_inventory.csv",index=False)
    es.to_csv(root/"center_error_summary.csv",index=False)
    a.to_csv(root/"audit_rows.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    ea.to_csv(root/"enrollment_audit.csv",index=False)
    a.to_csv(root/"fold_audit.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
        "outer_folds":nfold,"k_values":cfg["parameters"]["k_values"],
        "reserve_k":reserve_k,
        "fixed_downstream_pool":True,
        "seed_derivation":"sha256 first 8 bytes -> uint32",
        "formal_inference":"separate paired speaker-cluster bootstrap from OOF predictions",
        "enrollment":"nested prefix of one fixed 50-utterance label-free reserve per speaker"
    },indent=2)+"\n")

    print("CENTER ERROR"); print(es.to_string(index=False))
    print("\nSUMMARY"); print(summary.to_string(index=False))
    print("\nK DELTAS"); print(kd.to_string(index=False))

if __name__=="__main__":
    main()
