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
 "sample_id","dataset","speaker_id","annotator_count",
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

def weighted_ccc(y,p,w=None):
    y=np.asarray(y,float); p=np.asarray(p,float)
    if w is None: w=np.ones(len(y),float)
    w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def weighted_mae(y,p,w=None):
    if w is None: w=np.ones(len(y),float)
    w=np.asarray(w,float); w=w/w.sum()
    return float(np.sum(w*np.abs(np.asarray(y)-np.asarray(p))))

def prepare(raw,aset):
    d=raw[raw.dataset.eq("msp")].dropna(subset=[
        "sample_id","speaker_id","valence_mean_1_7","arousal_mean_1_7","dominance_mean_1_7"]).copy()
    d=d[d.speaker_id.astype(str).ne("Unknown")]
    if aset in ["pitch","all"]:
        d=d.dropna(subset=["f0_median_hz","pitch_relative_st","pitch_reference_scope"])
        d=d[(d.f0_median_hz>0)&d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float)); d["pitch_rel"]=d.pitch_relative_st.astype(float); d["pitch_base"]=d.pitch_abs-d.pitch_rel
    if aset in ["loudness","all"]:
        d=d.dropna(subset=["speech_lufs","loudness_relative_lu","loudness_reference_scope"])
        d=d[d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["loud_abs"]=d.speech_lufs.astype(float); d["loud_rel"]=d.loudness_relative_lu.astype(float); d["loud_base"]=d.loud_abs-d.loud_rel
    if aset in ["rate","all"]:
        d=d.dropna(subset=["phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"])
        d=d[(d.phoneme_articulation_rate>0)&(d.rate_relative_ratio>0)&d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float)); d["rate_rel"]=np.log(d.rate_relative_ratio.astype(float)); d["rate_base"]=d.rate_abs-d.rate_rel
    sizes=d.groupby("speaker_id").size()
    return d[d.speaker_id.isin(sizes[sizes>10].index)].sort_values("sample_id").reset_index(drop=True)

def attr_cols(aset,kind):
    roots={"pitch":("pitch_abs","pitch_rel","pitch_base"),
           "loudness":("loud_abs","loud_rel","loud_base"),
           "rate":("rate_abs","rate_rel","rate_base")}
    attrs=[aset] if aset!="all" else ["pitch","loudness","rate"]
    idx={"abs":0,"rel":1,"base":2}[kind]
    return [roots[a][idx] for a in attrs]

def fit_ridge(xtr,ytr,xte,wtr,alpha):
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    xtr=sc.transform(xtr); xte=sc.transform(xte)
    model=Ridge(alpha=alpha); model.fit(xtr,ytr,sample_weight=wtr)
    return model.predict(xte)

def bootstrap(v,reps,seed):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v))); m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]); root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)
    targets=cfg["parameters"]["targets"]; nfold=int(cfg["parameters"]["outer_folds"]); alpha=float(cfg["parameters"]["ridge_alpha"])
    var_rows=[]; between=[]; within=[]

    for aset in cfg["parameters"]["attribute_sets"]:
        d=prepare(raw,aset)
        # Oracle diagnostic target decomposition.
        for tname,tcol in targets.items():
            means=d.groupby("speaker_id")[tcol].mean()
            within_vars=d.groupby("speaker_id")[tcol].var(ddof=0)
            bv=float(means.var(ddof=0)); wv=float(within_vars.mean())
            var_rows.append({"attribute_set":aset,"target":tname,"between_variance":bv,"mean_within_variance":wv,
                             "icc_like":bv/(bv+wv) if bv+wv>0 else np.nan})
            d[f"{tname}_speaker_mean"]=d.speaker_id.map(means)
            d[f"{tname}_residual"]=d[tcol]-d[f"{tname}_speaker_mean"]

        speakers=sorted(d.speaker_id.astype(str).unique())
        # Speaker-level table.
        agg={}
        for c in attr_cols(aset,"base")+attr_cols(aset,"abs")+attr_cols(aset,"rel"):
            agg[c]="median"
        for tname in targets:
            agg[f"{tname}_speaker_mean"]="first"
        sp=d.groupby("speaker_id",as_index=False).agg(agg)

        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed))
            sf=d.speaker_id.astype(str).map(fmap).to_numpy()
            spf=sp.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                # Between-speaker speaker-mean prediction.
                trsp=spf!=fold; tesp=spf==fold
                ytr_sp=sp.loc[trsp,[f"{t}_speaker_mean" for t in targets]].to_numpy(float)
                yte_sp=sp.loc[tesp,[f"{t}_speaker_mean" for t in targets]].to_numpy(float)
                feature_sets={
                    "prosodic_baseline":attr_cols(aset,"base"),
                    "marginal_absolute":attr_cols(aset,"abs"),
                    "marginal_relative":attr_cols(aset,"rel"),
                    "baseline_plus_marginal_relative":attr_cols(aset,"base")+attr_cols(aset,"rel")
                }
                for rep,cols in feature_sets.items():
                    pred=fit_ridge(sp.loc[trsp,cols].to_numpy(np.float32),ytr_sp,sp.loc[tesp,cols].to_numpy(np.float32),
                                   np.ones(int(trsp.sum())),alpha)
                    for j,tname in enumerate(targets):
                        between.append({"attribute_set":aset,"seed":int(seed),"fold":fold,"representation":rep,"target":tname,
                                        "ccc":weighted_ccc(yte_sp[:,j],pred[:,j]),"mae":weighted_mae(yte_sp[:,j],pred[:,j]),
                                        "n_train_speakers":int(trsp.sum()),"n_test_speakers":int(tesp.sum())})

                # Within-speaker residual prediction.
                tr=sf!=fold; te=sf==fold
                train=d.loc[tr].copy(); test=d.loc[te].copy(); wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                ytr=d.loc[tr,[f"{t}_residual" for t in targets]].to_numpy(float)
                yte=d.loc[te,[f"{t}_residual" for t in targets]].to_numpy(float)
                reps={
                    "absolute":attr_cols(aset,"abs"),
                    "relative":attr_cols(aset,"rel"),
                    "baseline_only":attr_cols(aset,"base"),
                    "relative_plus_baseline":attr_cols(aset,"rel")+attr_cols(aset,"base")
                }
                for rep,cols in reps.items():
                    pred=fit_ridge(train[cols].to_numpy(np.float32),ytr,test[cols].to_numpy(np.float32),wtr,alpha)
                    for j,tname in enumerate(targets):
                        within.append({"attribute_set":aset,"seed":int(seed),"fold":fold,"representation":rep,"target":tname,
                                       "ccc":weighted_ccc(yte[:,j],pred[:,j],wte),"mae":weighted_mae(yte[:,j],pred[:,j],wte),
                                       "n_train":len(train),"n_test":len(test)})

    var=pd.DataFrame(var_rows); b=pd.DataFrame(between); w=pd.DataFrame(within)
    bs=b.groupby(["attribute_set","target","representation"],as_index=False).agg(ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),mae_mean=("mae","mean"),n_eval=("ccc","size"))
    ws=w.groupby(["attribute_set","target","representation"],as_index=False).agg(ccc_mean=("ccc","mean"),ccc_std=("ccc","std"),mae_mean=("mae","mean"),n_eval=("ccc","size"))
    p=w.pivot_table(index=["attribute_set","target","seed","fold"],columns="representation",values="ccc")
    rows=[]
    for (aset,target),g in p.groupby(level=[0,1]):
        for a,bn in [("relative","absolute"),("relative_plus_baseline","relative"),("baseline_only","relative")]:
            v=(g[a]-g[bn]).to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(aset,target,a,bn))
            rows.append({"attribute_set":aset,"target":target,"comparison":f"{a}-{bn}",
                         "delta_ccc_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)})
    wd=pd.DataFrame(rows)

    var.to_csv(root/"variance_decomposition.csv",index=False)
    b.to_csv(root/"between_speaker_metrics.csv",index=False)
    bs.to_csv(root/"between_speaker_summary.csv",index=False)
    w.to_csv(root/"within_speaker_metrics.csv",index=False)
    ws.to_csv(root/"within_speaker_summary.csv",index=False)
    wd.to_csv(root/"within_speaker_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],"outer_folds":nfold,
        "diagnostic_target_centering":"full-speaker VAD mean used only to define target components"
    },indent=2)+"\n")
    print("VARIANCE"); print(var.to_string(index=False))
    print("\nBETWEEN"); print(bs.to_string(index=False))
    print("\nWITHIN DELTAS"); print(wd.to_string(index=False))

if __name__=="__main__":
    main()
