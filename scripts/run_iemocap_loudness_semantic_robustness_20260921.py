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

def fit_predict(train,test,cols,ytr,wtr,alpha):
    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha); model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

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
    cols=["file","EmoAct","EmoVal","EmoDom","speaking_rate","pitch_mean","rms","relative_db"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)&(d.rms>0)].copy()
    d["speaker_id"]=d.file.map(speaker_id)
    d["sample_id"]=d.file.astype(str)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d["loud_source_abs"]=d.relative_db.astype(float)
    d["loud_rms_abs"]=20*np.log10(d.rms.astype(float))
    d=d.rename(columns={"EmoVal":"valence","EmoAct":"arousal","EmoDom":"dominance"})
    for t in ["valence","arousal","dominance"]:
        spm=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        d[f"{t}_spmean"]=d.speaker_id.map(spm)
        d[f"{t}_global"]=gm
        d[f"{t}_within"]=d[t]-d[f"{t}_spmean"]
    return d.reset_index(drop=True)

def add_reference(d,loud_abs_col,prefix):
    x=d.copy()
    source_cols=["pitch_abs",loud_abs_col,"rate_abs"]
    centers=x.groupby("speaker_id")[source_cols].median()
    for j,nm in enumerate(["pitch","loud","rate"]):
        x[f"{prefix}_{nm}_center"]=x.speaker_id.map(centers.iloc[:,j])
        x[f"{prefix}_{nm}_rel"]=x[source_cols[j]]-x[f"{prefix}_{nm}_center"]
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    base=prepare(cfg["dataset"]["iemocap_source_glob"])
    defs={
      "source_relative_db":("loud_source_abs","src"),
      "rms_db":("loud_rms_abs","rms")
    }
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    nfold=int(cfg["parameters"]["outer_folds"])
    speakers=sorted(base.speaker_id.astype(str).unique())
    rows=[]

    for definition,(loud_col,prefix) in defs.items():
        d=add_reference(base,loud_col,prefix)
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
                    sets={
                      "all":{
                        "absolute":["pitch_abs",loud_col,"rate_abs"],
                        "relative":[f"{prefix}_pitch_rel",f"{prefix}_loud_rel",f"{prefix}_rate_rel"]
                      },
                      "loudness":{
                        "absolute":[loud_col],
                        "relative":[f"{prefix}_loud_rel"]
                      }
                    }
                    for aset,reps in sets.items():
                        for rep,cols in reps.items():
                            pred=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                            for j,t in enumerate(targets):
                                rows.append({"definition":definition,"analysis_set":aset,
                                             "seed":int(seed),"fold":fold,"lambda":lam,
                                             "target":t,"representation":rep,
                                             "ccc":weighted_ccc(yte[:,j],pred[:,j],wte)})

    m=pd.DataFrame(rows)
    p=m.pivot_table(index=["definition","analysis_set","target","seed","fold","lambda"],
                    columns="representation",values="ccc").reset_index()
    p["relative_minus_absolute"]=p.relative-p.absolute
    curves=p.groupby(["definition","analysis_set","target","lambda"],as_index=False).agg(
        relative_minus_absolute_mean=("relative_minus_absolute","mean"))
    out=[]
    for definition in defs:
      for aset in ["all","loudness"]:
        for t in targets:
          g=p[(p.definition==definition)&(p.analysis_set==aset)&(p.target==t)]
          mat=g.pivot_table(index=["seed","fold"],columns="lambda",values="relative_minus_absolute")[lambdas].to_numpy()
          slope,lo,hi=bootstrap_slopes(mat,lambdas,int(cfg["parameters"]["bootstrap_reps"]),
                                       stable_seed(definition,aset,t))
          out.append({"definition":definition,"analysis_set":aset,"target":t,
                      "slope_mean":slope,"ci95_low":lo,"ci95_high":hi})
    slopes=pd.DataFrame(out)

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{
      "rows":len(base),"speakers":len(speakers),
      "rms_min":float(base.rms.min()),"rms_max":float(base.rms.max()),
      "corr_relative_db_vs_rms_db":float(np.corrcoef(base.loud_source_abs,base.loud_rms_abs)[0,1])
    }]).to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    curves.to_csv(root/"effect_curves.csv",index=False)
    slopes.to_csv(root/"slope_tests.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],
      "source_relative_db_semantics":"unknown from public dataset card",
      "rms_db_formula":"20*log10(rms)",
      "same_rows_across_definitions":True
    },indent=2)+"\n")
    print("SLOPES"); print(slopes.to_string(index=False))

if __name__=="__main__":
    main()
