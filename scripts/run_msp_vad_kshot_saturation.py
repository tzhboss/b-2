#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
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
 "phoneme_articulation_rate","rate_reference_scope"
]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def enrollment_indices(df,k,seed):
    out=[]
    for sp,idx in df.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        rng=np.random.default_rng(stable_seed(seed,sp,k,"saturation-enrollment"))
        out.extend(rng.choice(idx,size=k,replace=False).tolist())
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
    model=Ridge(alpha=alpha); model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]); root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(src,columns=COLS)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    abs_cols=["pitch_abs","loud_abs","rate_abs"]
    sizes=d.groupby("speaker_id").size()
    keep=sizes[sizes>int(cfg["parameters"]["min_support_exclusive"])].index
    d=d[d.speaker_id.isin(keep)].sort_values("sample_id").reset_index(drop=True)
    speakers=sorted(d.speaker_id.astype(str).unique())
    marginal={sp:g[abs_cols].median().to_numpy(float) for sp,g in d.groupby("speaker_id")}
    target_cols=list(cfg["parameters"]["targets"].values())
    nfold=int(cfg["parameters"]["outer_folds"])
    metrics=[]; errors=[]

    for seed in cfg["seed"]:
        fmap=speaker_folds(speakers,nfold,int(seed)); sf=d.speaker_id.astype(str).map(fmap).to_numpy()
        for k in cfg["parameters"]["k_values"]:
            enroll=enrollment_indices(d,int(k),stable_seed("exp20",seed))
            em=np.zeros(len(d),bool); em[enroll]=True
            kest={sp:g[abs_cols].median().to_numpy(float) for sp,g in d.loc[em].groupby("speaker_id")}
            for fold in range(nfold):
                tr=(sf!=fold)&(~em); te=(sf==fold)&(~em)
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                for part in [train,test]:
                    kc=np.vstack([kest[sp] for sp in part.speaker_id])
                    mc=np.vstack([marginal[sp] for sp in part.speaker_id])
                    av=part[abs_cols].to_numpy(float)
                    for j,name in enumerate(["pitch","loud","rate"]):
                        part[f"{name}_k_base"]=kc[:,j]; part[f"{name}_k_rel"]=av[:,j]-kc[:,j]
                        part[f"{name}_marg_base"]=mc[:,j]; part[f"{name}_marg_rel"]=av[:,j]-mc[:,j]
                for sp in sorted(set(test.speaker_id.astype(str))):
                    for j,name in enumerate(["pitch","loudness","log_rate"]):
                        errors.append({"seed":int(seed),"fold":fold,"K":int(k),"speaker_id":sp,"attribute":name,
                                       "error_to_marginal":abs(float(kest[sp][j])-float(marginal[sp][j]))})
                reps={
                    "absolute":abs_cols,
                    "kshot_hybrid":["pitch_k_rel","loud_k_rel","rate_k_rel","pitch_k_base","loud_k_base","rate_k_base"],
                    "marginal_oracle_hybrid":["pitch_marg_rel","loud_marg_rel","rate_marg_rel","pitch_marg_base","loud_marg_base","rate_marg_base"]
                }
                ytr=train[target_cols].to_numpy(float); yte=test[target_cols].to_numpy(float)
                wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                for rep,cols in reps.items():
                    pred=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                    for j,tname in enumerate(cfg["parameters"]["targets"]):
                        metrics.append({"seed":int(seed),"fold":fold,"K":int(k),"representation":rep,"target":tname,
                                        "ccc":weighted_ccc(yte[:,j],pred[:,j],wte)})

    m=pd.DataFrame(metrics); e=pd.DataFrame(errors)
    es=e.groupby(["K","attribute"],as_index=False).agg(mae_to_marginal=("error_to_marginal","mean"))
    s=m.groupby(["K","target","representation"],as_index=False).agg(ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),n_eval=("ccc","size"))
    rows=[]
    p=s.pivot_table(index=["target","representation"],columns="K",values="ccc_mean")
    for target in cfg["parameters"]["targets"]:
        for rep in ["kshot_hybrid","absolute","marginal_oracle_hybrid"]:
            row={"target":target,"representation":rep}
            for a,b in [(1,10),(10,20),(20,50),(10,50)]:
                row[f"delta_K{b}_minus_K{a}"]=float(p.loc[(target,rep),b]-p.loc[(target,rep),a])
            rows.append(row)
    kd=pd.DataFrame(rows)

    pd.DataFrame([{"rows":len(d),"speakers":len(speakers),"min_rows_per_speaker":int(d.groupby("speaker_id").size().min())}]).to_csv(root/"data_inventory.csv",index=False)
    es.to_csv(root/"center_error_summary.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    s.to_csv(root/"summary.csv",index=False)
    kd.to_csv(root/"k_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
      "outer_folds":nfold,"k_values":cfg["parameters"]["k_values"],
      "fixed_cohort_min_support_exclusive":cfg["parameters"]["min_support_exclusive"]
    },indent=2)+"\n")
    print("CENTER ERROR"); print(es.to_string(index=False))
    print("\nSUMMARY"); print(s.to_string(index=False))
    print("\nK DELTAS"); print(kd.to_string(index=False))

if __name__=="__main__":
    main()
