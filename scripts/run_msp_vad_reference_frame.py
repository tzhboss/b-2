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
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(df):
    counts=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/counts[s]).to_numpy(float)
    return w / w.mean()

def weighted_stats(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float)
    w=w/w.sum()
    my=float(np.sum(w*y)); mp=float(np.sum(w*p))
    dy=y-my; dp=p-mp
    vy=float(np.sum(w*dy*dy)); vp=float(np.sum(w*dp*dp)); cov=float(np.sum(w*dy*dp))
    ccc=(2*cov)/(vy+vp+(my-mp)**2) if (vy+vp+(my-mp)**2)>0 else np.nan
    pearson=cov/np.sqrt(vy*vp) if vy>0 and vp>0 else np.nan
    mae=float(np.sum(w*np.abs(y-p)))
    rmse=float(np.sqrt(np.sum(w*(y-p)**2)))
    r2=1.0-float(np.sum(w*(y-p)**2))/vy if vy>0 else np.nan
    return ccc,pearson,mae,rmse,r2

def prepare(raw,aset):
    d=raw[raw.dataset.eq("msp")].copy()
    d=d.dropna(subset=["sample_id","speaker_id","valence_mean_1_7","arousal_mean_1_7","dominance_mean_1_7"])
    d=d[d.speaker_id.astype(str).ne("Unknown")].copy()

    req=[]
    if aset in ["pitch","all"]:
        req += ["f0_median_hz","pitch_relative_st","pitch_reference_scope"]
    if aset in ["loudness","all"]:
        req += ["speech_lufs","loudness_relative_lu","loudness_reference_scope"]
    if aset in ["rate","all"]:
        req += ["phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"]
    d=d.dropna(subset=req).copy()

    if aset in ["pitch","all"]:
        d=d[(d.f0_median_hz>0)&d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
        d["pitch_rel"]=d.pitch_relative_st.astype(float)
        d["pitch_base"]=d.pitch_abs-d.pitch_rel
    if aset in ["loudness","all"]:
        d=d[d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["loud_abs"]=d.speech_lufs.astype(float)
        d["loud_rel"]=d.loudness_relative_lu.astype(float)
        d["loud_base"]=d.loud_abs-d.loud_rel
    if aset in ["rate","all"]:
        d=d[(d.phoneme_articulation_rate>0)&(d.rate_relative_ratio>0)&
            d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])]
        d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
        d["rate_rel"]=np.log(d.rate_relative_ratio.astype(float))
        d["rate_base"]=d.rate_abs-d.rate_rel

    sizes=d.groupby("speaker_id").size()
    d=d[d.speaker_id.isin(sizes[sizes>10].index)].sort_values("sample_id").reset_index(drop=True)
    return d

def rep_columns(aset,rep):
    roots={"pitch":("pitch_abs","pitch_rel","pitch_base"),
           "loudness":("loud_abs","loud_rel","loud_base"),
           "rate":("rate_abs","rate_rel","rate_base")}
    attrs=[aset] if aset!="all" else ["pitch","loudness","rate"]
    abs_cols=[roots[a][0] for a in attrs]
    rel_cols=[roots[a][1] for a in attrs]
    base_cols=[roots[a][2] for a in attrs]
    if rep=="absolute": return abs_cols
    if rep=="relative": return rel_cols
    if rep=="baseline_only": return base_cols
    if rep=="relative_plus_baseline": return rel_cols+base_cols
    raise ValueError(rep)

def bootstrap(v,reps,seed):
    v=np.asarray(v,float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),size=(reps,len(v)))
    m=v[idx].mean(1)
    return tuple(map(float,np.quantile(m,[.025,.975])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)
    targets=cfg["parameters"]["targets"]
    metrics=[]; inv=[]
    nfold=int(cfg["parameters"]["outer_folds"])

    for aset in cfg["parameters"]["attribute_sets"]:
        d=prepare(raw,aset)
        speakers=sorted(d.speaker_id.astype(str).unique())
        inv.append({
            "attribute_set":aset,"rows":len(d),"speakers":len(speakers),
            "min_rows_per_speaker":int(d.groupby("speaker_id").size().min()),
            "mean_annotator_count":float(d.annotator_count.mean()),
            "unknown_rows":int((d.speaker_id.astype(str)=="Unknown").sum())
        })
        Y=d[list(targets.values())].to_numpy(float)
        for seed in cfg["seed"]:
            fmap=speaker_folds(speakers,nfold,int(seed))
            sf=d.speaker_id.astype(str).map(fmap).to_numpy()
            for fold in range(nfold):
                tr=sf!=fold; te=sf==fold
                train=d.loc[tr].copy(); test=d.loc[te].copy()
                if set(train.speaker_id.astype(str))&set(test.speaker_id.astype(str)):
                    raise RuntimeError("speaker leakage")
                wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
                ytr=Y[tr]; yte=Y[te]
                for rep in cfg["parameters"]["representations"]:
                    cols=rep_columns(aset,rep)
                    xtr=train[cols].to_numpy(np.float32); xte=test[cols].to_numpy(np.float32)
                    scaler=StandardScaler()
                    scaler.fit(xtr,sample_weight=wtr)
                    xtr=scaler.transform(xtr); xte=scaler.transform(xte)
                    model=Ridge(alpha=float(cfg["parameters"]["ridge_alpha"]))
                    model.fit(xtr,ytr,sample_weight=wtr)
                    pred=model.predict(xte)
                    for j,(tname,_) in enumerate(targets.items()):
                        ccc,pr,mae,rmse,r2=weighted_stats(yte[:,j],pred[:,j],wte)
                        metrics.append({
                            "attribute_set":aset,"seed":int(seed),"fold":fold,"representation":rep,
                            "target":tname,"speaker_balanced_ccc":ccc,"speaker_balanced_pearson":pr,
                            "speaker_balanced_mae":mae,"speaker_balanced_rmse":rmse,"speaker_balanced_r2":r2,
                            "n_train":len(train),"n_test":len(test),
                            "train_speakers":train.speaker_id.nunique(),"test_speakers":test.speaker_id.nunique()
                        })

    m=pd.DataFrame(metrics); inventory=pd.DataFrame(inv)
    summary=m.groupby(["attribute_set","target","representation"],as_index=False).agg(
        ccc_mean=("speaker_balanced_ccc","mean"),ccc_std=("speaker_balanced_ccc","std"),
        pearson_mean=("speaker_balanced_pearson","mean"),
        mae_mean=("speaker_balanced_mae","mean"),rmse_mean=("speaker_balanced_rmse","mean"),
        r2_mean=("speaker_balanced_r2","mean"),n_eval=("speaker_balanced_ccc","size"))
    p=m.pivot_table(index=["attribute_set","target","seed","fold"],columns="representation",values="speaker_balanced_ccc")
    rows=[]
    for (aset,target),g in p.groupby(level=[0,1]):
        for a,b in [("relative","absolute"),("relative_plus_baseline","relative"),
                    ("baseline_only","absolute"),("relative_plus_baseline","absolute")]:
            v=(g[a]-g[b]).dropna().to_numpy()
            lo,hi=bootstrap(v,int(cfg["parameters"]["bootstrap_reps"]),stable_seed(aset,target,a,b))
            rows.append({
                "attribute_set":aset,"target":target,"comparison":f"{a}-{b}",
                "delta_ccc_mean":float(v.mean()),"ci95_low":lo,"ci95_high":hi,"n_pairs":len(v)
            })
    deltas=pd.DataFrame(rows)

    inventory.to_csv(root/"data_inventory.csv",index=False)
    m.to_csv(root/"metrics_by_fold.csv",index=False)
    summary.to_csv(root/"summary.csv",index=False)
    deltas.to_csv(root/"paired_deltas.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"source":str(src),"seeds":cfg["seed"],
        "outer_folds":nfold,"ridge_alpha":cfg["parameters"]["ridge_alpha"],
        "vad_source":"MSP labels_detailed.csv human SAM 1-7 ratings aggregated by utterance mean",
        "speaker_weighting":"equal total weight per speaker in train and evaluation"
    },indent=2)+"\n")
    print("SUMMARY"); print(summary.to_string(index=False))
    print("\nPAIRED DELTAS"); print(deltas.to_string(index=False))

if __name__=="__main__":
    main()
