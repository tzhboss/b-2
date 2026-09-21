#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

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

def iemocap_speaker(x):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x))
    if not m: raise ValueError(x)
    return m.group(1)+'_'+m.group(2)

def prepare_msp(path):
    cols=["dataset","sample_id","speaker_id","valence_mean_1_7","arousal_mean_1_7","dominance_mean_1_7",
          "f0_median_hz","speech_lufs","phoneme_articulation_rate",
          "pitch_reference_scope","loudness_reference_scope","rate_reference_scope"]
    d=pd.read_parquet(path,columns=cols)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d=d.rename(columns={"valence_mean_1_7":"valence","arousal_mean_1_7":"arousal","dominance_mean_1_7":"dominance"})
    return d[["sample_id","speaker_id","valence","arousal","dominance","pitch_abs","loud_abs","rate_abs"]].reset_index(drop=True)

def prepare_iemocap(pattern):
    files=sorted(glob.glob(pattern))
    cols=["file","EmoAct","EmoVal","EmoDom","speaking_rate","pitch_mean","relative_db"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str)
    d["speaker_id"]=d.file.map(iemocap_speaker)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["loud_abs"]=d.relative_db.astype(float)
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoVal":"valence","EmoAct":"arousal","EmoDom":"dominance"})
    return d[["sample_id","speaker_id","valence","arousal","dominance","pitch_abs","loud_abs","rate_abs"]].reset_index(drop=True)

def add_target_components(d):
    for t in ["valence","arousal","dominance"]:
        spmean=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        d[f"{t}_spmean"]=d.speaker_id.map(spmean)
        d[f"{t}_global"]=gm
        d[f"{t}_within"]=d[t]-d[f"{t}_spmean"]
    return d

def nested_enrollment(d,k,seed):
    chosen=[]
    for sp,idx in d.groupby("speaker_id",sort=True).indices.items():
        idx=np.asarray(idx,dtype=int)
        if len(idx)<=k:
            raise RuntimeError(f"{sp}: only {len(idx)} rows for K={k}")
        rng=np.random.default_rng(stable_seed(seed,sp,"kshot-target-reference"))
        perm=idx.copy(); rng.shuffle(perm)
        chosen.extend(perm[:k].tolist())
    return np.asarray(sorted(chosen),dtype=int)

def bootstrap_slopes(mat,lambdas,reps,seed):
    mat=np.asarray(mat,float); lambdas=np.asarray(lambdas,float)
    slopes=np.array([np.polyfit(lambdas,row,1)[0] for row in mat])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(slopes),size=(reps,len(slopes)))
    bs=slopes[idx].mean(1)
    lo,hi=np.quantile(bs,[.025,.975])
    return float(slopes.mean()),float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    k=int(cfg["parameters"]["k_enrollment"])
    datasets={
      "msp":add_target_components(prepare_msp(Path(os.environ[cfg["dataset"]["msp_source_env"]]))),
      "iemocap":add_target_components(prepare_iemocap(cfg["dataset"]["iemocap_source_glob"]))
    }
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    abs_cols=["pitch_abs","loud_abs","rate_abs"]
    nfold=int(cfg["parameters"]["outer_folds"])
    rows=[]; audits=[]; inv=[]

    for ds,d0 in datasets.items():
        sizes=d0.groupby("speaker_id").size()
        d=d0[d0.speaker_id.isin(sizes[sizes>k].index)].copy().reset_index(drop=True)
        speakers=sorted(d.speaker_id.astype(str).unique())
        inv.append({"dataset":ds,"rows_pre_enrollment":len(d),"speakers":len(speakers),
                    "min_support":int(d.groupby("speaker_id").size().min())})
        for seed in cfg["seed"]:
            enroll_idx=nested_enrollment(d,k,int(seed))
            em=np.zeros(len(d),bool); em[enroll_idx]=True
            center=d.loc[em].groupby("speaker_id")[abs_cols].median()
            if len(center)!=len(speakers): raise RuntimeError("missing speaker center")
            work=d.loc[~em].copy().reset_index(drop=True)
            for j,name in enumerate(["pitch","loud","rate"]):
                work[f"{name}_center"]=work.speaker_id.map(center.iloc[:,j])
                work[f"{name}_rel"]=work[abs_cols[j]]-work[f"{name}_center"]
            fmap=speaker_folds(speakers,nfold,int(seed)); sf=work.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                tr=sf!=fold; te=sf==fold
                train=work.loc[tr].copy(); test=work.loc[te].copy()
                if set(train.speaker_id)&set(test.speaker_id): raise RuntimeError("speaker overlap")
                wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                audits.append({"dataset":ds,"seed":int(seed),"fold":fold,
                               "train_speakers":train.speaker_id.nunique(),
                               "test_speakers":test.speaker_id.nunique(),
                               "n_train":len(train),"n_test":len(test),
                               "enrollment_rows":int(em.sum())})
                reps={
                    "absolute":abs_cols,
                    "kshot_relative":["pitch_rel","loud_rel","rate_rel"],
                    "kshot_hybrid":["pitch_rel","loud_rel","rate_rel","pitch_center","loud_center","rate_center"]
                }
                for lam in lambdas:
                    ytr=np.column_stack([
                        train[f"{t}_global"].to_numpy(float)+train[f"{t}_within"].to_numpy(float)+
                        lam*(train[f"{t}_spmean"].to_numpy(float)-train[f"{t}_global"].to_numpy(float))
                        for t in targets])
                    yte=np.column_stack([
                        test[f"{t}_global"].to_numpy(float)+test[f"{t}_within"].to_numpy(float)+
                        lam*(test[f"{t}_spmean"].to_numpy(float)-test[f"{t}_global"].to_numpy(float))
                        for t in targets])
                    for rep,cols in reps.items():
                        pred=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                        for jj,t in enumerate(targets):
                            rows.append({"dataset":ds,"seed":int(seed),"fold":fold,"lambda":lam,
                                         "target":t,"representation":rep,
                                         "ccc":weighted_ccc(yte[:,jj],pred[:,jj],wte)})

    m=pd.DataFrame(rows)
    p=m.pivot_table(index=["dataset","target","seed","fold","lambda"],columns="representation",values="ccc").reset_index()
    p["kshot_relative_minus_absolute"]=p["kshot_relative"]-p["absolute"]
    p["kshot_hybrid_minus_relative"]=p["kshot_hybrid"]-p["kshot_relative"]
    curves=p.groupby(["dataset","target","lambda"],as_index=False).agg(
        kshot_relative_minus_absolute_mean=("kshot_relative_minus_absolute","mean"),
        kshot_hybrid_minus_relative_mean=("kshot_hybrid_minus_relative","mean"))
    out=[]
    for ds in datasets:
        for t in targets:
            g=p[(p.dataset==ds)&(p.target==t)]
            for effect in ["kshot_relative_minus_absolute","kshot_hybrid_minus_relative"]:
                mat=g.pivot_table(index=["seed","fold"],columns="lambda",values=effect)[lambdas].to_numpy()
                slope,lo,hi=bootstrap_slopes(mat,lambdas,int(cfg["parameters"]["bootstrap_reps"]),
                                             stable_seed(ds,t,effect))
                out.append({"dataset":ds,"target":t,"effect":effect,
                            "slope_mean":slope,"ci95_low":lo,"ci95_high":hi})
    slopes=pd.DataFrame(out)
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(inv).to_csv(root/"data_inventory.csv",index=False)
    pd.DataFrame(audits).to_csv(root/"audit_rows.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    curves.to_csv(root/"effect_curves.csv",index=False)
    slopes.to_csv(root/"slope_tests.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"k_enrollment":k,"lambda_values":lambdas,
      "reference_estimation":"per-speaker median from K unlabeled enrollment utterances; enrollment excluded from downstream rows"
    },indent=2)+"\n")
    print("CURVES"); print(curves.to_string(index=False))
    print("\nSLOPES"); print(slopes.to_string(index=False))

if __name__=="__main__":
    main()
