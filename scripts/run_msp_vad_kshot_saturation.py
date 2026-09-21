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
 "f0_median_hz","pitch_relative_st","pitch_reference_scope",
 "speech_lufs","loudness_relative_lu","loudness_reference_scope",
 "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"
]

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object); rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def enrollment_indices(df,k,seed):
    out=[]
    for sp,idx in df.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        if len(idx)<=k: raise RuntimeError(f"{sp}: only {len(idx)} rows for K={k}")
        rng=np.random.default_rng(stable_seed(seed,sp,"vad-nested-enrollment"))
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
    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha); model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]); root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(src,columns=COLS)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&(d.rate_relative_ratio>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    sizes=d.groupby("speaker_id").size()
    min_support=int(cfg["parameters"]["min_support"])
    d=d[d.speaker_id.isin(sizes[sizes>=min_support].index)].sort_values("sample_id").reset_index(drop=True)

    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d["pitch_neutral_rel"]=d.pitch_relative_st.astype(float)
    d["loud_neutral_rel"]=d.loudness_relative_lu.astype(float)
    d["rate_neutral_rel"]=np.log(d.rate_relative_ratio.astype(float))
    d["pitch_neutral_base"]=d.pitch_abs-d.pitch_neutral_rel
    d["loud_neutral_base"]=d.loud_abs-d.loud_neutral_rel
    d["rate_neutral_base"]=d.rate_abs-d.rate_neutral_rel

    abs_cols=["pitch_abs","loud_abs","rate_abs"]
    target_cols=list(cfg["parameters"]["targets"].values())
    speakers=sorted(d.speaker_id.astype(str).unique()); nfold=int(cfg["parameters"]["outer_folds"])
    # Full marginal diagnostic oracle center.
    marginal={sp:g[abs_cols].median().to_numpy(float) for sp,g in d.groupby("speaker_id")}
    neutral={sp:g[["pitch_neutral_base","loud_neutral_base","rate_neutral_base"]].median().to_numpy(float) for sp,g in d.groupby("speaker_id")}

    metrics=[]; errors=[]; audits=[]
    for seed in cfg["seed"]:
        fmap=speaker_folds(speakers,nfold,int(seed)); sf=d.speaker_id.astype(str).map(fmap).to_numpy()
        for k in cfg["parameters"]["k_values"]:
            enroll=enrollment_indices(d,int(k),stable_seed("msp-vad",seed))
            em=np.zeros(len(d),bool); em[enroll]=True
            kest={sp:g[abs_cols].median().to_numpy(float) for sp,g in d.loc[em].groupby("speaker_id")}
            for fold in range(nfold):
                tr=(sf!=fold)&(~em); te=(sf==fold)&(~em)
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                train_s=set(train.speaker_id.astype(str)); test_s=set(test.speaker_id.astype(str))
                if train_s&test_s: raise RuntimeError("speaker leakage")
                if np.any(em&(tr|te)): raise RuntimeError("enrollment overlap")
                for part in [train,test]:
                    kc=np.vstack([kest[sp] for sp in part.speaker_id])
                    mc=np.vstack([marginal[sp] for sp in part.speaker_id])
                    nc=np.vstack([neutral[sp] for sp in part.speaker_id])
                    av=part[abs_cols].to_numpy(float)
                    for j,name in enumerate(["pitch","loud","rate"]):
                        part[f"{name}_k_base"]=kc[:,j]; part[f"{name}_k_rel"]=av[:,j]-kc[:,j]
                        part[f"{name}_marg_base"]=mc[:,j]; part[f"{name}_marg_rel"]=av[:,j]-mc[:,j]
                        part[f"{name}_neu_base"]=nc[:,j]; part[f"{name}_neu_rel"]=av[:,j]-nc[:,j]

                for sp in sorted(test_s):
                    for j,name in enumerate(["pitch","loudness","log_rate"]):
                        errors.append({"seed":int(seed),"fold":fold,"K":int(k),"speaker_id":sp,"attribute":name,
                                       "error_to_marginal":abs(float(kest[sp][j])-float(marginal[sp][j])),
                                       "error_to_neutral":abs(float(kest[sp][j])-float(neutral[sp][j]))})

                reps={
                  "absolute":abs_cols,
                  "kshot_relative":["pitch_k_rel","loud_k_rel","rate_k_rel"],
                  "kshot_hybrid":["pitch_k_rel","loud_k_rel","rate_k_rel","pitch_k_base","loud_k_base","rate_k_base"],
                  "neutral_oracle_relative":["pitch_neu_rel","loud_neu_rel","rate_neu_rel"],
                  "neutral_oracle_hybrid":["pitch_neu_rel","loud_neu_rel","rate_neu_rel","pitch_neu_base","loud_neu_base","rate_neu_base"],
                  "marginal_oracle_relative":["pitch_marg_rel","loud_marg_rel","rate_marg_rel"],
                  "marginal_oracle_hybrid":["pitch_marg_rel","loud_marg_rel","rate_marg_rel","pitch_marg_base","loud_marg_base","rate_marg_base"]
                }
                ytr=train[target_cols].to_numpy(float); yte=test[target_cols].to_numpy(float); wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                for rep,cols in reps.items():
                    pred=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                    for j,tname in enumerate(cfg["parameters"]["targets"]):
                        metrics.append({"seed":int(seed),"fold":fold,"K":int(k),"representation":rep,"target":tname,
                                        "ccc":weighted_ccc(yte[:,j],pred[:,j],wte),"n_train":len(train),"n_test":len(test)})
                audits.append({"seed":int(seed),"fold":fold,"K":int(k),"train_speakers":len(train_s),"test_speakers":len(test_s),
                               "enrollment_eval_overlap":int(np.any(em&(tr|te))),"speaker_overlap":len(train_s&test_s)})

    m=pd.DataFrame(metrics); e=pd.DataFrame(errors); a=pd.DataFrame(audits)
    summary=m.groupby(["K","target","representation"],as_index=False).agg(ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),n_eval=("ccc","size"))
    es=e.groupby(["K","attribute"],as_index=False).agg(mae_to_marginal=("error_to_marginal","mean"),mae_to_neutral=("error_to_neutral","mean"),n_speaker_evals=("speaker_id","size"))
    p=m.pivot_table(index=["K","target","seed","fold"],columns="representation",values="ccc")
    rows=[]
    comps=[("kshot_hybrid","absolute"),("kshot_relative","absolute"),("kshot_hybrid","marginal_oracle_hybrid"),
           ("kshot_hybrid","neutral_oracle_hybrid"),("marginal_oracle_hybrid","absolute"),("neutral_oracle_hybrid","absolute")]
    for (k,target),g in p.groupby(level=[0,1]):
        for x,y in comps:
            v=(g[x]-g[y]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(k,target,x,y))
            rows.append({"K":int(k),"target":target,"comparison":f"{x}-{y}","delta_ccc_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    delta=pd.DataFrame(rows)

    pd.DataFrame([{"rows":len(d),"speakers":len(speakers),"min_rows_per_speaker":int(d.groupby("speaker_id").size().min())}]).to_csv(root/"data_inventory.csv",index=False)
    e.to_csv(root/"center_error_by_speaker.csv",index=False)
    es.to_csv(root/"center_error_summary.csv",index=False)
    a.to_csv(root/"audit_rows.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    delta.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],"outer_folds":nfold,
      "k_values":cfg["parameters"]["k_values"],"common_min_support":cfg["parameters"]["min_support"],
      "enrollment":"single nested random order per speaker, no VAD labels",
      "marginal_oracle":"diagnostic full eligible speaker median"
    },indent=2)+"\n")
    print("CENTER ERROR"); print(es.to_string(index=False))
    print("\nSUMMARY"); print(summary.to_string(index=False))
    print("\nDELTAS"); print(delta.to_string(index=False))

if __name__=="__main__":
    main()
