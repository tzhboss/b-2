#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_id(name):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(name))
    if not m:
        raise ValueError(name)
    return m.group(1)+'_'+m.group(2)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(df):
    c=df.speaker_id.value_counts()
    w=df.speaker_id.map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def weighted_mae(y,p,w):
    w=np.asarray(w,float); w=w/w.sum()
    return float(np.sum(w*np.abs(np.asarray(y)-np.asarray(p))))

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
    files=sorted(glob.glob(cfg["dataset"]["source_glob"]))
    cols=["file","EmoAct","EmoVal","EmoDom","speaking_rate","pitch_mean","relative_db"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True)
    d=d.dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["speaker_id"]=d.file.map(speaker_id)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["loud_abs"]=d.relative_db.astype(float)
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    abs_cols=["pitch_abs","loud_abs","rate_abs"]

    centers=d.groupby("speaker_id")[abs_cols].median()
    for j,name in enumerate(["pitch","loud","rate"]):
        d[f"{name}_base"]=d.speaker_id.map(centers.iloc[:,j])
        d[f"{name}_rel"]=d[abs_cols[j]]-d[f"{name}_base"]

    targets=cfg["parameters"]["targets"]
    var_rows=[]
    for tname,tcol in targets.items():
        means=d.groupby("speaker_id")[tcol].mean()
        within=d.groupby("speaker_id")[tcol].var(ddof=0)
        bv=float(means.var(ddof=0)); wv=float(within.mean())
        d[f"{tname}_mean"]=d.speaker_id.map(means)
        d[f"{tname}_resid"]=d[tcol]-d[f"{tname}_mean"]
        var_rows.append({"target":tname,"between_variance":bv,"mean_within_variance":wv,
                         "icc_like":bv/(bv+wv) if bv+wv>0 else np.nan})

    reps={
      "absolute":abs_cols,
      "relative":["pitch_rel","loud_rel","rate_rel"],
      "baseline_only":["pitch_base","loud_base","rate_base"],
      "relative_plus_baseline":["pitch_rel","loud_rel","rate_rel","pitch_base","loud_base","rate_base"]
    }
    speakers=sorted(d.speaker_id.unique())
    rows=[]
    nfold=int(cfg["parameters"]["outer_folds"])
    target_names=list(targets)
    for seed in cfg["seed"]:
        fmap=speaker_folds(speakers,nfold,int(seed)); sf=d.speaker_id.map(fmap).to_numpy()
        for fold in range(nfold):
            tr=sf!=fold; te=sf==fold
            train=d.loc[tr].copy(); test=d.loc[te].copy()
            wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
            for task in ["raw","residual"]:
                ytr=np.column_stack([
                    train[targets[t]].to_numpy(float) if task=="raw" else train[f"{t}_resid"].to_numpy(float)
                    for t in target_names])
                yte=np.column_stack([
                    test[targets[t]].to_numpy(float) if task=="raw" else test[f"{t}_resid"].to_numpy(float)
                    for t in target_names])
                for rep,cols2 in reps.items():
                    pred=fit_predict(train,test,cols2,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                    for j,t in enumerate(target_names):
                        rows.append({"seed":int(seed),"fold":fold,"task":task,"target":t,"representation":rep,
                                     "ccc":weighted_ccc(yte[:,j],pred[:,j],wte),
                                     "mae":weighted_mae(yte[:,j],pred[:,j],wte),
                                     "n_train":len(train),"n_test":len(test),
                                     "train_speakers":train.speaker_id.nunique(),"test_speakers":test.speaker_id.nunique()})

    m=pd.DataFrame(rows)
    s=m.groupby(["task","target","representation"],as_index=False).agg(
        ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),mae_mean=("mae","mean"),n_eval=("ccc","size"))
    p=m.pivot_table(index=["task","target","seed","fold"],columns="representation",values="ccc")
    out=[]
    for (task,target),g in p.groupby(level=[0,1]):
        for a,b in [("relative_plus_baseline","relative"),("relative","absolute"),
                    ("relative_plus_baseline","absolute")]:
            v=(g[a]-g[b]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(task,target,a,b))
            out.append({"task":task,"target":target,"comparison":f"{a}-{b}",
                        "delta_ccc_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    delta=pd.DataFrame(out)

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"rows":len(d),"speakers":len(speakers),"min_rows_per_speaker":int(d.groupby("speaker_id").size().min()),
                   "max_rows_per_speaker":int(d.groupby("speaker_id").size().max())}]).to_csv(root/"data_inventory.csv",index=False)
    pd.DataFrame(var_rows).to_csv(root/"variance_decomposition.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    s.to_csv(root/"summary.csv",index=False)
    delta.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source_files":files,"seeds":cfg["seed"],
      "outer_folds":nfold,"speaker_id_rule":"session ID + utterance F/M suffix",
      "feature_mapping":{"pitch":"12log2(pitch_mean)","loudness":"relative_db","rate":"log(speaking_rate)"},
      "diagnostic_oracle":"full-speaker acoustic median and VAD mean"
    },indent=2)+"\n")
    print("VARIANCE"); print(pd.DataFrame(var_rows).to_string(index=False))
    print("\nSUMMARY"); print(s.to_string(index=False))
    print("\nDELTAS"); print(delta.to_string(index=False))

if __name__=="__main__":
    main()
