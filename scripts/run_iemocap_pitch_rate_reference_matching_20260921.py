#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_id(x):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x))
    if not m: raise ValueError(x)
    return m.group(1)+'_'+m.group(2)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def fit_predict(train,test,cols,ytr,wtr,alpha,family):
    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    xtr=sc.transform(xtr); xte=sc.transform(xte)
    if family=="quadratic_ridge":
        poly=PolynomialFeatures(degree=2,include_bias=False)
        xtr=poly.fit_transform(xtr); xte=poly.transform(xte)
        sc2=StandardScaler(); sc2.fit(xtr,sample_weight=wtr)
        xtr=sc2.transform(xtr); xte=sc2.transform(xte)
    model=Ridge(alpha=alpha); model.fit(xtr,ytr,sample_weight=wtr)
    return model.predict(xte)

def bootstrap_slopes(mat,lambdas,reps,seed):
    mat=np.asarray(mat,float); lambdas=np.asarray(lambdas,float)
    slopes=np.array([np.polyfit(lambdas,row,1)[0] for row in mat])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(slopes),size=(reps,len(slopes)))
    bs=slopes[idx].mean(1)
    lo,hi=np.quantile(bs,[.025,.975])
    return float(slopes.mean()),float(lo),float(hi)

def prepare(pattern):
    files=sorted(glob.glob(pattern))
    cols=["file","EmoAct","EmoVal","EmoDom","speaking_rate","pitch_mean"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str)
    d["speaker_id"]=d.file.map(speaker_id)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoVal":"valence","EmoAct":"arousal","EmoDom":"dominance"})
    for a in ["pitch","rate"]:
        ctr=d.groupby("speaker_id")[f"{a}_abs"].median()
        d[f"{a}_center"]=d.speaker_id.map(ctr)
        d[f"{a}_rel"]=d[f"{a}_abs"]-d[f"{a}_center"]
    for t in ["valence","arousal","dominance"]:
        spm=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        d[f"{t}_spmean"]=d.speaker_id.map(spm)
        d[f"{t}_global"]=gm
        d[f"{t}_within"]=d[t]-d[f"{t}_spmean"]
    return d.reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    d=prepare(cfg["dataset"]["iemocap_source_glob"])
    speakers=sorted(d.speaker_id.astype(str).unique())
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    families=list(cfg["parameters"]["model_families"])
    nfold=int(cfg["parameters"]["outer_folds"])
    sets={
      "pitch":["pitch"],
      "rate":["rate"],
      "pitch_rate":["pitch","rate"]
    }
    rows=[]
    for seed in cfg["seed"]:
        fmap=speaker_folds(speakers,nfold,int(seed)); sf=d.speaker_id.astype(str).map(fmap).to_numpy()
        for fold in range(nfold):
            tr=sf!=fold; te=sf==fold
            train=d.loc[tr].copy(); test=d.loc[te].copy()
            wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
            for lam in lambdas:
                ytr=np.column_stack([
                    train[f"{t}_global"].to_numpy(float)+train[f"{t}_within"].to_numpy(float)+
                    lam*(train[f"{t}_spmean"].to_numpy(float)-train[f"{t}_global"].to_numpy(float))
                    for t in targets])
                yte=np.column_stack([
                    test[f"{t}_global"].to_numpy(float)+test[f"{t}_within"].to_numpy(float)+
                    lam*(test[f"{t}_spmean"].to_numpy(float)-test[f"{t}_global"].to_numpy(float))
                    for t in targets])
                for aset,attrs in sets.items():
                    reps={
                      "absolute":[f"{a}_abs" for a in attrs],
                      "relative":[f"{a}_rel" for a in attrs],
                      "hybrid":[f"{a}_rel" for a in attrs]+[f"{a}_center" for a in attrs]
                    }
                    for family in families:
                        for rep,cols in reps.items():
                            pred=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]),family)
                            for j,t in enumerate(targets):
                                rows.append({"seed":int(seed),"fold":fold,"lambda":lam,
                                             "feature_set":aset,"model_family":family,
                                             "target":t,"representation":rep,
                                             "ccc":weighted_ccc(yte[:,j],pred[:,j],wte)})
    m=pd.DataFrame(rows)
    p=m.pivot_table(index=["feature_set","model_family","target","seed","fold","lambda"],
                    columns="representation",values="ccc").reset_index()
    p["relative_minus_absolute"]=p.relative-p.absolute
    p["hybrid_minus_relative"]=p.hybrid-p.relative
    curves=p.groupby(["feature_set","model_family","target","lambda"],as_index=False).agg(
        relative_minus_absolute_mean=("relative_minus_absolute","mean"),
        hybrid_minus_relative_mean=("hybrid_minus_relative","mean"))
    out=[]
    for aset in sets:
      for family in families:
        for t in targets:
          g=p[(p.feature_set==aset)&(p.model_family==family)&(p.target==t)]
          for effect in ["relative_minus_absolute","hybrid_minus_relative"]:
            mat=g.pivot_table(index=["seed","fold"],columns="lambda",values=effect)[lambdas].to_numpy()
            slope,lo,hi=bootstrap_slopes(mat,lambdas,int(cfg["parameters"]["bootstrap_reps"]),
                                         stable_seed(aset,family,t,effect))
            out.append({"feature_set":aset,"model_family":family,"target":t,"effect":effect,
                        "slope_mean":slope,"ci95_low":lo,"ci95_high":hi})
    slopes=pd.DataFrame(out)
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"rows":len(d),"speakers":len(speakers),
                   "pitch_min_hz":float(d.pitch_mean.min()),
                   "rate_min":float(d.speaking_rate.min())}]).to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    curves.to_csv(root/"effect_curves.csv",index=False)
    slopes.to_csv(root/"slope_tests.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],
      "excluded_fields":["relative_db","rms"],
      "feature_sets":list(sets),"model_families":families
    },indent=2)+"\n")
    print("SLOPES"); print(slopes.to_string(index=False))

if __name__=="__main__":
    main()
