#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

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
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float)
    if len(y)==0: return np.nan
    w=w/w.sum()
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

def iemocap_speaker(x):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x))
    if not m: raise ValueError(x)
    return m.group(1)+'_'+m.group(2)

def prepare_msp(path):
    cols=["dataset","sample_id","speaker_id","arousal_mean_1_7","dominance_mean_1_7",
          "f0_median_hz","phoneme_articulation_rate","pitch_reference_scope","rate_reference_scope"]
    d=pd.read_parquet(path,columns=cols)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    sizes=d.groupby("speaker_id").size(); d=d[d.speaker_id.isin(sizes[sizes>10].index)].copy()
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d=d.rename(columns={"arousal_mean_1_7":"arousal","dominance_mean_1_7":"dominance"})
    return add_components(d[["sample_id","speaker_id","arousal","dominance","pitch_abs","rate_abs"]].reset_index(drop=True))

def prepare_iemocap(pattern):
    files=sorted(glob.glob(pattern))
    cols=["file","EmoAct","EmoDom","speaking_rate","pitch_mean"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in files],ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(iemocap_speaker)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})
    return add_components(d[["sample_id","speaker_id","arousal","dominance","pitch_abs","rate_abs"]].reset_index(drop=True))

def add_components(d):
    for a in ["pitch","rate"]:
        ctr=d.groupby("speaker_id")[f"{a}_abs"].median()
        d[f"{a}_center"]=d.speaker_id.map(ctr)
        d[f"{a}_rel"]=d[f"{a}_abs"]-d[f"{a}_center"]
    for t in ["arousal","dominance"]:
        spm=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        d[f"{t}_spmean"]=d.speaker_id.map(spm)
        d[f"{t}_global"]=gm
        d[f"{t}_within"]=d[t]-d[f"{t}_spmean"]
    return d

def cluster_effect(frame, target, lam):
    g=frame[(frame.target==target)&(frame["lambda"]==lam)]
    # one sampled speaker occurrence has equal total mass
    counts=g.groupby("cluster_instance").size()
    w=g.cluster_instance.map(lambda s:1.0/counts[s]).to_numpy(float)
    return weighted_ccc(g.y_true,g.pred_relative,w)-weighted_ccc(g.y_true,g.pred_absolute,w)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    datasets={
      "msp":prepare_msp(Path(os.environ[cfg["dataset"]["msp_source_env"]])),
      "iemocap":prepare_iemocap(cfg["dataset"]["iemocap_source_glob"])
    }
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    families=list(cfg["parameters"]["model_families"])
    nfold=int(cfg["parameters"]["outer_folds"]); split_seed=int(cfg["parameters"]["split_seed"])
    reps={"absolute":["pitch_abs","rate_abs"],"relative":["pitch_rel","rate_rel"]}
    oof=[]; inv=[]
    for ds,d in datasets.items():
        speakers=sorted(d.speaker_id.astype(str).unique())
        inv.append({"dataset":ds,"rows":len(d),"speakers":len(speakers)})
        fmap=speaker_folds(speakers,nfold,split_seed); sf=d.speaker_id.astype(str).map(fmap).to_numpy()
        for fold in range(nfold):
            tr=sf!=fold; te=sf==fold
            train=d.loc[tr].copy(); test=d.loc[te].copy()
            wtr=equal_speaker_weights(train)
            for lam in lambdas:
                ytr=np.column_stack([
                    train[f"{t}_global"].to_numpy(float)+train[f"{t}_within"].to_numpy(float)+
                    lam*(train[f"{t}_spmean"].to_numpy(float)-train[f"{t}_global"].to_numpy(float))
                    for t in targets])
                yte=np.column_stack([
                    test[f"{t}_global"].to_numpy(float)+test[f"{t}_within"].to_numpy(float)+
                    lam*(test[f"{t}_spmean"].to_numpy(float)-test[f"{t}_global"].to_numpy(float))
                    for t in targets])
                for fam in families:
                    preds={}
                    for rep,cols in reps.items():
                        preds[rep]=fit_predict(train,test,cols,ytr,wtr,float(cfg["parameters"]["ridge_alpha"]),fam)
                    for j,t in enumerate(targets):
                        for ii,rowidx in enumerate(test.index):
                            oof.append({"dataset":ds,"speaker_id":str(test.loc[rowidx,"speaker_id"]),
                                        "sample_id":str(test.loc[rowidx,"sample_id"]),"fold":fold,
                                        "lambda":lam,"model_family":fam,"target":t,
                                        "y_true":float(yte[ii,j]),
                                        "pred_absolute":float(preds["absolute"][ii,j]),
                                        "pred_relative":float(preds["relative"][ii,j])})
    o=pd.DataFrame(oof)
    points=[]
    boots=[]
    B=int(cfg["parameters"]["bootstrap_reps"])
    for ds in datasets:
      dd=o[o.dataset==ds].copy()
      speakers=np.array(sorted(dd.speaker_id.unique()),dtype=object)
      for fam in families:
        ff=dd[dd.model_family==fam]
        for t in targets:
          effects=[]
          for lam in lambdas:
            g=ff[(ff.target==t)&(ff["lambda"]==lam)].copy()
            counts=g.groupby("speaker_id").size()
            w=g.speaker_id.map(lambda s:1.0/counts[s]).to_numpy(float)
            eff=weighted_ccc(g.y_true,g.pred_relative,w)-weighted_ccc(g.y_true,g.pred_absolute,w)
            effects.append(eff)
            points.append({"dataset":ds,"model_family":fam,"target":t,"lambda":lam,
                           "relative_minus_absolute":eff})
          point_slope=float(np.polyfit(lambdas,effects,1)[0])
          rng=np.random.default_rng(stable_seed(ds,fam,t,"speaker_bootstrap"))
          bs_slopes=np.empty(B,float)
          bs_l0=np.empty(B,float)
          for b in range(B):
            samp=rng.choice(speakers,size=len(speakers),replace=True)
            pieces=[]
            for occ,sp in enumerate(samp):
                z=ff[(ff.target==t)&(ff.speaker_id==sp)].copy()
                z["cluster_instance"]=occ
                pieces.append(z)
            q=pd.concat(pieces,ignore_index=True)
            ev=[cluster_effect(q,t,lam) for lam in lambdas]
            bs_slopes[b]=np.polyfit(lambdas,ev,1)[0]
            bs_l0[b]=ev[0]
          slo,shi=np.quantile(bs_slopes,[.025,.975])
          elo,ehi=np.quantile(bs_l0,[.025,.975])
          boots.append({"dataset":ds,"model_family":fam,"target":t,
                        "slope_mean":point_slope,"slope_ci95_low":float(slo),"slope_ci95_high":float(shi),
                        "lambda0_effect":effects[0],"lambda0_ci95_low":float(elo),"lambda0_ci95_high":float(ehi),
                        "speakers":len(speakers),"bootstrap_reps":B})
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(inv).to_csv(root/"data_inventory.csv",index=False)
    o.to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(points).to_csv(root/"point_effects.csv",index=False)
    pd.DataFrame(boots).to_csv(root/"cluster_bootstrap_slopes.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"split_seed":split_seed,"outer_folds":nfold,
      "bootstrap_unit":"speaker","bootstrap_reps":B
    },indent=2)+"\n")
    print(pd.DataFrame(boots).to_string(index=False))

if __name__=="__main__":
    main()
