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
    my=np.sum(w*y); mp=np.sum(w*p); dy=y-my; dp=p-mp
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
    sizes=d.groupby("speaker_id").size(); d=d[d.speaker_id.isin(sizes[sizes>10].index)].copy()
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["loud_abs"]=d.speech_lufs.astype(float); d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d=d.rename(columns={"valence_mean_1_7":"valence","arousal_mean_1_7":"arousal","dominance_mean_1_7":"dominance"})
    return d[["sample_id","speaker_id","valence","arousal","dominance","pitch_abs","loud_abs","rate_abs"]].reset_index(drop=True)

def prepare_iemocap(pattern):
    files=sorted(glob.glob(pattern))
    cols=["file","EmoAct","EmoVal","EmoDom","speaking_rate","pitch_mean","relative_db"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(iemocap_speaker)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["loud_abs"]=d.relative_db.astype(float); d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoVal":"valence","EmoAct":"arousal","EmoDom":"dominance"})
    return d[["sample_id","speaker_id","valence","arousal","dominance","pitch_abs","loud_abs","rate_abs"]].reset_index(drop=True)

def add_components(d):
    abs_cols=["pitch_abs","loud_abs","rate_abs"]
    centers=d.groupby("speaker_id")[abs_cols].median()
    for j,name in enumerate(["pitch","loud","rate"]):
        d[f"{name}_center"]=d.speaker_id.map(centers.iloc[:,j])
        d[f"{name}_rel"]=d[abs_cols[j]]-d[f"{name}_center"]
    for t in ["valence","arousal","dominance"]:
        spmean=d.groupby("speaker_id")[t].mean()
        d[f"{t}_spmean"]=d.speaker_id.map(spmean)
        d[f"{t}_resid"]=d[t]-d[f"{t}_spmean"]
    return d, centers

def derangement_map(speakers, centers, seed):
    speakers=list(sorted(map(str,speakers)))
    if len(speakers)<2: raise RuntimeError("need >=2 speakers")
    rng=np.random.default_rng(seed)
    perm=speakers.copy()
    for _ in range(1000):
        rng.shuffle(perm)
        if all(a!=b for a,b in zip(speakers,perm)):
            break
    else:
        perm=speakers[1:]+speakers[:1]
    return {sp:centers.loc[src].to_numpy(float) for sp,src in zip(speakers,perm)}

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); means=v[idx].mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    return float(v.mean()),float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    datasets={}
    for name,d in {
      "msp":prepare_msp(Path(os.environ[cfg["dataset"]["msp_source_env"]])),
      "iemocap":prepare_iemocap(cfg["dataset"]["iemocap_source_glob"])
    }.items():
        datasets[name]=add_components(d)

    targets=list(cfg["parameters"]["targets"]); tasks=list(cfg["parameters"]["tasks"])
    nfold=int(cfg["parameters"]["outer_folds"]); nperm=int(cfg["parameters"]["permutation_reps"])
    rel_cols=["pitch_rel","loud_rel","rate_rel"]
    center_cols=["pitch_center","loud_center","rate_center"]
    metrics=[]; inv=[]

    for ds,(d,centers) in datasets.items():
        speakers=sorted(d.speaker_id.astype(str).unique())
        inv.append({"dataset":ds,"rows":len(d),"speakers":len(speakers)})
        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed)); sf=d.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                tr=sf!=fold; te=sf==fold
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                train_s=sorted(train.speaker_id.astype(str).unique())
                test_s=sorted(test.speaker_id.astype(str).unique())
                wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                for task in tasks:
                    ytr=np.column_stack([train[t].to_numpy(float) if task=="raw" else train[f"{t}_resid"].to_numpy(float) for t in targets])
                    yte=np.column_stack([test[t].to_numpy(float) if task=="raw" else test[f"{t}_resid"].to_numpy(float) for t in targets])

                    pred_rel=fit_predict(train,test,rel_cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                    pred_true=fit_predict(train,test,rel_cols+center_cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                    for j,t in enumerate(targets):
                        metrics.append({"dataset":ds,"seed":int(seed),"fold":fold,"task":task,"target":t,
                                        "condition":"relative","perm_rep":-1,"ccc":weighted_ccc(yte[:,j],pred_rel[:,j],wte)})
                        metrics.append({"dataset":ds,"seed":int(seed),"fold":fold,"task":task,"target":t,
                                        "condition":"true_hybrid","perm_rep":-1,"ccc":weighted_ccc(yte[:,j],pred_true[:,j],wte)})

                    for rep in range(nperm):
                        trmap=derangement_map(train_s,centers,stable_seed(ds,seed,fold,task,rep,"train"))
                        temap=derangement_map(test_s,centers,stable_seed(ds,seed,fold,task,rep,"test"))
                        trp=train.copy(); tep=test.copy()
                        trcent=np.vstack([trmap[str(sp)] for sp in trp.speaker_id])
                        tecent=np.vstack([temap[str(sp)] for sp in tep.speaker_id])
                        for jj,nm in enumerate(["pitch","loud","rate"]):
                            trp[f"{nm}_perm_center"]=trcent[:,jj]
                            tep[f"{nm}_perm_center"]=tecent[:,jj]
                        pcols=rel_cols+["pitch_perm_center","loud_perm_center","rate_perm_center"]
                        pred=fit_predict(trp,tep,pcols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]))
                        for j,t in enumerate(targets):
                            metrics.append({"dataset":ds,"seed":int(seed),"fold":fold,"task":task,"target":t,
                                            "condition":"permuted_hybrid","perm_rep":rep,
                                            "ccc":weighted_ccc(yte[:,j],pred[:,j],wte)})

    m=pd.DataFrame(metrics)
    true=m[m.condition=="true_hybrid"][["dataset","seed","fold","task","target","ccc"]].rename(columns={"ccc":"true_ccc"})
    perm=m[m.condition=="permuted_hybrid"].groupby(["dataset","seed","fold","task","target"],as_index=False).ccc.mean().rename(columns={"ccc":"mean_permuted_ccc"})
    e=true.merge(perm,on=["dataset","seed","fold","task","target"],validate="one_to_one")
    e["true_minus_permuted"]=e.true_ccc-e.mean_permuted_ccc
    out=[]
    for (ds,task,t),g in e.groupby(["dataset","task","target"]):
        mean,lo,hi=bootstrap(g.true_minus_permuted.to_numpy(),int(cfg["parameters"]["bootstrap_reps"]),stable_seed(ds,task,t))
        out.append({"dataset":ds,"task":task,"target":t,"delta_mean":mean,"ci95_low":lo,"ci95_high":hi,"n_pairs":len(g)})
    summary=pd.DataFrame(out)

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(inv).to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    e.to_csv(root/"permutation_effects_by_fold.csv",index=False)
    summary.to_csv(root/"effect_summary.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"permutation_reps":nperm,
      "intervention":"keep true Relative features; derange appended speaker center identity separately in train/test"
    },indent=2)+"\n")
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
