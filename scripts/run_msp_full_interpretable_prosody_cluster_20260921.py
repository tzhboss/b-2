#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
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
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p); dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def fit_predict(xtr,xte,ytr,wtr,alpha,fam):
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    xtr=sc.transform(xtr); xte=sc.transform(xte)
    if fam=="quadratic_ridge":
        poly=PolynomialFeatures(degree=2,include_bias=False)
        xtr=poly.fit_transform(xtr); xte=poly.transform(xte)
        sc2=StandardScaler(); sc2.fit(xtr,sample_weight=wtr)
        xtr=sc2.transform(xtr); xte=sc2.transform(xte)
    m=Ridge(alpha=alpha); m.fit(xtr,ytr,sample_weight=wtr)
    return m.predict(xte)

def ccc_from_moments(a):
    my=a[:,0]; ey2=a[:,1]; mp=a[:,2]; ep2=a[:,3]; eyp=a[:,4]
    vy=ey2-my*my; vp=ep2-mp*mp; cov=eyp-my*mp
    den=vy+vp+(my-mp)**2
    out=np.full(len(my),np.nan,float); ok=den>0
    out[ok]=2*cov[ok]/den[ok]
    return out

def per_speaker_moments(g,speakers,predcol):
    rows=[]
    for sp in speakers:
        z=g[g.speaker_id==sp]
        y=z.y_true.to_numpy(float); p=z[predcol].to_numpy(float)
        rows.append([y.mean(),np.mean(y*y),p.mean(),np.mean(p*p),np.mean(y*p)])
    return np.asarray(rows,float)

def bootstrap_effects(g,speakers,lambdas,pred_a,pred_b,B,seed):
    stats={}
    for lam in lambdas:
        z=g[g["lambda"]==lam]
        stats[lam]=(per_speaker_moments(z,speakers,pred_a),per_speaker_moments(z,speakers,pred_b))
    S=len(speakers); rng=np.random.default_rng(seed)
    pvec=np.full(S,1.0/S)
    x=np.asarray(lambdas,float); xc=x-x.mean(); den=float(xc@xc)
    all_eff=[]; off=0
    while off<B:
        n=min(250,B-off)
        counts=rng.multinomial(S,pvec,size=n).astype(float)/S
        eff=[]
        for lam in lambdas:
            a,b=stats[lam]
            ca=ccc_from_moments(counts@a)
            cb=ccc_from_moments(counts@b)
            eff.append(ca-cb)
        eff=np.column_stack(eff)
        all_eff.append(eff); off+=n
    E=np.vstack(all_eff)
    slopes=(E@xc)/den
    return E,slopes

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    cols=["dataset","sample_id","speaker_id","arousal_mean_1_7","dominance_mean_1_7",
          "f0_median_hz","f0_iqr_semitone","speech_lufs","phoneme_articulation_rate",
          "pause_ratio","voiced_ratio","pitch_reference_scope","loudness_reference_scope","rate_reference_scope"]
    d=pd.read_parquet(Path(os.environ[cfg["dataset"]["source_env"]]),columns=cols)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[(d.speaker_id.astype(str)!="Unknown")&(d.f0_median_hz>0)&(d.phoneme_articulation_rate>0)&
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])&
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
    sizes=d.groupby("speaker_id").size(); d=d[d.speaker_id.isin(sizes[sizes>10].index)].copy()
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["f0var_abs"]=d.f0_iqr_semitone.astype(float)
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d["pause_abs"]=d.pause_ratio.astype(float)
    d["voiced_abs"]=d.voiced_ratio.astype(float)
    d=d.rename(columns={"arousal_mean_1_7":"arousal","dominance_mean_1_7":"dominance"})
    attrs=["pitch","f0var","loud","rate","pause","voiced"]
    for a in attrs:
        ctr=d.groupby("speaker_id")[f"{a}_abs"].median()
        d[f"{a}_center"]=d.speaker_id.map(ctr)
        d[f"{a}_rel"]=d[f"{a}_abs"]-d[f"{a}_center"]
    for t in ["arousal","dominance"]:
        spm=d.groupby("speaker_id")[t].mean(); gm=float(d[t].mean())
        d[f"{t}_spmean"]=d.speaker_id.map(spm); d[f"{t}_global"]=gm; d[f"{t}_within"]=d[t]-d[f"{t}_spmean"]

    abs_cols=[f"{a}_abs" for a in attrs]
    rel_cols=[f"{a}_rel" for a in attrs]
    center_cols=[f"{a}_center" for a in attrs]
    reps={"absolute":abs_cols,"relative":rel_cols,"hybrid":rel_cols+center_cols}
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"]); fams=list(cfg["parameters"]["model_families"])
    speakers=sorted(d.speaker_id.astype(str).unique())
    fmap=speaker_folds(speakers,int(cfg["parameters"]["outer_folds"]),int(cfg["parameters"]["split_seed"]))
    sf=d.speaker_id.astype(str).map(fmap).to_numpy()
    oof=[]

    for fold in range(int(cfg["parameters"]["outer_folds"])):
        tr=sf!=fold; te=sf==fold
        train=d.loc[tr].copy(); test=d.loc[te].copy()
        wtr=equal_speaker_weights(train)
        for lam in lambdas:
            Ytr=np.column_stack([
                train[f"{t}_global"].to_numpy(float)+train[f"{t}_within"].to_numpy(float)+
                lam*(train[f"{t}_spmean"].to_numpy(float)-train[f"{t}_global"].to_numpy(float))
                for t in targets])
            Yte=np.column_stack([
                test[f"{t}_global"].to_numpy(float)+test[f"{t}_within"].to_numpy(float)+
                lam*(test[f"{t}_spmean"].to_numpy(float)-test[f"{t}_global"].to_numpy(float))
                for t in targets])
            for fam in fams:
                preds={}
                for rep,cols2 in reps.items():
                    preds[rep]=fit_predict(train[cols2].to_numpy(np.float32),test[cols2].to_numpy(np.float32),
                                           Ytr,wtr,float(cfg["parameters"]["ridge_alpha"]),fam)
                test_idx=np.flatnonzero(te)
                for j,t in enumerate(targets):
                    for pos,ri in enumerate(test_idx):
                        oof.append({"sample_id":d.iloc[ri].sample_id,"speaker_id":str(d.iloc[ri].speaker_id),
                                    "fold":fold,"lambda":lam,"model_family":fam,"target":t,
                                    "y_true":float(Yte[pos,j]),
                                    "pred_absolute":float(preds["absolute"][pos,j]),
                                    "pred_relative":float(preds["relative"][pos,j]),
                                    "pred_hybrid":float(preds["hybrid"][pos,j])})
    o=pd.DataFrame(oof)
    B=int(cfg["parameters"]["bootstrap_reps"])
    summaries=[]; points=[]
    for fam in fams:
      for t in targets:
        g=o[(o.model_family==fam)&(o.target==t)]
        # point effects with equal speaker weight
        for lam in lambdas:
            z=g[g["lambda"]==lam]; counts=z.groupby("speaker_id").size()
            w=z.speaker_id.map(lambda s:1.0/counts[s]).to_numpy(float)
            ca=weighted_ccc(z.y_true,z.pred_absolute,w)
            cr=weighted_ccc(z.y_true,z.pred_relative,w)
            ch=weighted_ccc(z.y_true,z.pred_hybrid,w)
            points.append({"model_family":fam,"target":t,"lambda":lam,
                           "relative_minus_absolute":cr-ca,
                           "hybrid_minus_relative":ch-cr,
                           "hybrid_minus_absolute":ch-ca})
        E,sl=bootstrap_effects(g,speakers,lambdas,"pred_relative","pred_absolute",B,stable_seed(fam,t,"ra"))
        H,sh=bootstrap_effects(g,speakers,lambdas,"pred_hybrid","pred_relative",B,stable_seed(fam,t,"hr"))
        psub=[x for x in points if x["model_family"]==fam and x["target"]==t]
        psub=sorted(psub,key=lambda x:x["lambda"])
        point_slope=float(np.polyfit(lambdas,[x["relative_minus_absolute"] for x in psub],1)[0])
        lo,hi=np.quantile(sl,[.025,.975])
        l0lo,l0hi=np.quantile(E[:,0],[.025,.975])
        h1lo,h1hi=np.quantile(H[:,-1],[.025,.975])
        summaries.append({"model_family":fam,"target":t,
                          "ra_slope":point_slope,"ra_slope_ci95_low":float(lo),"ra_slope_ci95_high":float(hi),
                          "ra_lambda0":psub[0]["relative_minus_absolute"],
                          "ra_lambda0_ci95_low":float(l0lo),"ra_lambda0_ci95_high":float(l0hi),
                          "hybrid_minus_relative_lambda1":psub[-1]["hybrid_minus_relative"],
                          "hr_lambda1_ci95_low":float(h1lo),"hr_lambda1_ci95_high":float(h1hi)})
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"rows":len(d),"speakers":len(speakers),"features":len(attrs)}]).to_csv(root/"data_inventory.csv",index=False)
    o.to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(points).to_csv(root/"point_effects.csv",index=False)
    pd.DataFrame(summaries).to_csv(root/"cluster_bootstrap_summary.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"attributes":attrs,"bootstrap_unit":"speaker","bootstrap_reps":B
    },indent=2)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=="__main__":
    main()
