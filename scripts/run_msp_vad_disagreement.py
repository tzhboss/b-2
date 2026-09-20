#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_COLS=[
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
    src=Path(os.environ[cfg["dataset"]["source_env"]]); agr=Path(os.environ[cfg["dataset"]["agreement_env"]])
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=BASE_COLS)
    acols=["sample_id"]+[v["std"] for v in cfg["parameters"]["targets"].values()]
    agreement=pd.read_parquet(agr,columns=acols)

    d=raw[raw.dataset.eq("msp")].merge(agreement,on="sample_id",how="inner",validate="one_to_one")
    need=["speaker_id","valence_mean_1_7","arousal_mean_1_7","dominance_mean_1_7",
          "f0_median_hz","pitch_relative_st","pitch_reference_scope",
          "speech_lufs","loudness_relative_lu","loudness_reference_scope",
          "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"]+[v["std"] for v in cfg["parameters"]["targets"].values()]
    d=d.dropna(subset=need).copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&
        (d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&(d.rate_relative_ratio>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    sizes=d.groupby("speaker_id").size(); d=d[d.speaker_id.isin(sizes[sizes>10].index)].sort_values("sample_id").reset_index(drop=True)

    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float)); d["pitch_rel"]=d.pitch_relative_st.astype(float); d["pitch_base"]=d.pitch_abs-d.pitch_rel
    d["loud_abs"]=d.speech_lufs.astype(float); d["loud_rel"]=d.loudness_relative_lu.astype(float); d["loud_base"]=d.loud_abs-d.loud_rel
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float)); d["rate_rel"]=np.log(d.rate_relative_ratio.astype(float)); d["rate_base"]=d.rate_abs-d.rate_rel
    reps={
      "absolute":["pitch_abs","loud_abs","rate_abs"],
      "relative":["pitch_rel","loud_rel","rate_rel"],
      "relative_plus_baseline":["pitch_rel","loud_rel","rate_rel","pitch_base","loud_base","rate_base"]
    }
    targets=cfg["parameters"]["targets"]
    # Diagnostic residual labels.
    for tname,tmeta in targets.items():
        means=d.groupby("speaker_id")[tmeta["mean"]].mean()
        d[f"{tname}_residual"]=d[tmeta["mean"]]-d.speaker_id.map(means)

    speakers=sorted(d.speaker_id.astype(str).unique())
    metrics=[]; thresholds=[]
    nfold=int(cfg["parameters"]["outer_folds"]); qlo,qhi=cfg["parameters"]["quantiles"]
    for seed in cfg["seed"]:
        fmap=speaker_folds(speakers,nfold,int(seed)); sf=d.speaker_id.astype(str).map(fmap).to_numpy()
        for fold in range(nfold):
            train=d.loc[sf!=fold].copy(); test=d.loc[sf==fold].copy(); wtr=equal_speaker_weights(train)
            pred_cache={}
            for task in cfg["parameters"]["tasks"]:
                ytr=np.column_stack([
                    train[tmeta["mean"]].to_numpy(float) if task=="raw" else train[f"{tname}_residual"].to_numpy(float)
                    for tname,tmeta in targets.items()])
                for rep,cols in reps.items():
                    pred_cache[(task,rep)]=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))

            for j,(tname,tmeta) in enumerate(targets.items()):
                lo=float(train[tmeta["std"]].quantile(qlo)); hi=float(train[tmeta["std"]].quantile(qhi))
                thresholds.append({"seed":int(seed),"fold":fold,"target":tname,"low_threshold":lo,"high_threshold":hi})
                std=test[tmeta["std"]].to_numpy(float)
                masks={"low":std<=lo,"mid":(std>lo)&(std<hi),"high":std>=hi,"all":np.ones(len(test),bool)}
                for stratum,mask in masks.items():
                    if mask.sum()<10: continue
                    sub=test.loc[mask]; wte=equal_speaker_weights(sub)
                    for task in cfg["parameters"]["tasks"]:
                        y=test[tmeta["mean"]].to_numpy(float) if task=="raw" else test[f"{tname}_residual"].to_numpy(float)
                        for rep in reps:
                            pred=pred_cache[(task,rep)][:,j]
                            metrics.append({"seed":int(seed),"fold":fold,"target":tname,"task":task,"stratum":stratum,
                                            "representation":rep,"ccc":weighted_ccc(y[mask],pred[mask],wte),
                                            "n_test":int(mask.sum()),"test_speakers":sub.speaker_id.nunique()})

    m=pd.DataFrame(metrics); th=pd.DataFrame(thresholds)
    summary=m.groupby(["target","task","stratum","representation"],as_index=False).agg(
        ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),n_eval=("ccc","size"),mean_n_test=("n_test","mean"))
    p=m.pivot_table(index=["target","task","stratum","seed","fold"],columns="representation",values="ccc")
    rows=[]
    for (target,task,stratum),g in p.groupby(level=[0,1,2]):
        for a,b in [("relative_plus_baseline","relative"),("relative","absolute")]:
            v=(g[a]-g[b]).dropna().to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(target,task,stratum,a,b))
            rows.append({"target":target,"task":task,"stratum":stratum,"comparison":f"{a}-{b}",
                         "delta_ccc_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    deltas=pd.DataFrame(rows)

    pd.DataFrame([{"rows":len(d),"speakers":d.speaker_id.nunique(),"min_rows_per_speaker":int(d.groupby("speaker_id").size().min()),
                   "annotation_match_rows":len(d)}]).to_csv(root/"data_inventory.csv",index=False)
    th.to_csv(root/"stratum_thresholds.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"agreement_source":str(agr),
      "quantiles":cfg["parameters"]["quantiles"],"seeds":cfg["seed"],"outer_folds":nfold,
      "threshold_policy":"train-only per-target disagreement quantiles"
    },indent=2)+"\n")
    print("SUMMARY"); print(summary.to_string(index=False))
    print("\nDELTAS"); print(deltas.to_string(index=False))

if __name__=="__main__":
    main()
