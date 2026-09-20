#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLS=[
 "sample_id","dataset","speaker_id","emotion_5class_candidate","emotion_training_usable",
 "f0_median_hz","pitch_relative_st","pitch_reference_scope",
 "speech_lufs","loudness_relative_lu","loudness_reference_scope",
 "phoneme_articulation_rate","rate_relative_ratio","rate_reference_scope"
]

ATTRS={
 "pitch": dict(abs="f0_median_hz",rel="pitch_relative_st",scope="pitch_reference_scope",abs_tf="pitch_st",rel_tf="identity",cabs=.5,crel=.5),
 "loudness": dict(abs="speech_lufs",rel="loudness_relative_lu",scope="loudness_reference_scope",abs_tf="identity",rel_tf="identity",cabs=1.0,crel=1.0),
 "rate": dict(abs="phoneme_articulation_rate",rel="rate_relative_ratio",scope="rate_reference_scope",abs_tf="log",rel_tf="log",cabs=.05,crel=.05),
}
NUMERIC=[
 "class_prevalence","speaker_coverage","absolute_effect_signed","relative_effect_signed",
 "baseline_effect_signed","absolute_effect_abs","relative_effect_abs","baseline_effect_abs",
 "relative_minus_absolute_separation","baseline_prior_corr",
 "between_baseline_to_within_relative_sd_ratio","conflict_fraction"
]
CAT=["attribute","emotion"]

def transform(x,kind):
    x=x.astype(float)
    if kind=="identity": return x
    if kind=="log":
        if not bool((x>0).all()): raise RuntimeError("nonpositive log input")
        return np.log(x)
    if kind=="pitch_st":
        if not bool((x>0).all()): raise RuntimeError("nonpositive pitch")
        return 12*np.log2(x)
    raise ValueError(kind)

def prepare(raw,dataset,attr):
    a=ATTRS[attr]
    d=raw[raw.dataset.eq(dataset)].copy()
    d=d[d.emotion_training_usable.fillna(False)&d.emotion_5class_candidate.fillna("").ne("")].copy()
    d=d.dropna(subset=["speaker_id",a["abs"],a["rel"],a["scope"]])
    if dataset in ["msp","meld"]:
        d=d[d[a["scope"]].isin(["speaker_neutral","speaker_neutral_shrunk"])].copy()
        d=d[d.speaker_id.astype(str).ne("Unknown")].copy()
    d["abs_v"]=transform(d[a["abs"]],a["abs_tf"])
    d["rel_v"]=transform(d[a["rel"]],a["rel_tf"])
    d=d[np.isfinite(d.abs_v)&np.isfinite(d.rel_v)].copy()
    d["base_v"]=d.abs_v-d.rel_v
    if dataset in ["msp","meld"]:
        sz=d.groupby("speaker_id").size()
        d=d[d.speaker_id.isin(sz[sz>10].index)].copy()
    return d

def weighted_mean_var(x,w):
    x=np.asarray(x,float); w=np.asarray(w,float)
    w=w/w.sum()
    m=float(np.sum(w*x))
    v=float(np.sum(w*(x-m)**2))
    return m,v

def weighted_effect(d,label,col):
    y=d.emotion_5class_candidate.astype(str).eq(label).to_numpy()
    counts=d.speaker_id.astype(str).value_counts()
    w=d.speaker_id.astype(str).map(lambda s:1.0/counts[s]).to_numpy(float)
    m1,v1=weighted_mean_var(d.loc[y,col],w[y])
    m0,v0=weighted_mean_var(d.loc[~y,col],w[~y])
    pooled=np.sqrt(max((v1+v0)/2,1e-12))
    return float((m1-m0)/pooled)

def safe_corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12: return 0.0
    return float(np.corrcoef(x,y)[0,1])

def feature_row(d,dataset,attr,emotion):
    a=ATTRS[attr]
    n=len(d)
    cls=d.emotion_5class_candidate.astype(str).eq(emotion)
    prev=float(cls.mean())
    sp_total=d.speaker_id.nunique()
    sp_cov=float(d.loc[cls,"speaker_id"].nunique()/sp_total)
    ae=weighted_effect(d,emotion,"abs_v")
    re=weighted_effect(d,emotion,"rel_v")
    be=weighted_effect(d,emotion,"base_v")
    sp=d.groupby("speaker_id").agg(
        baseline=("base_v","median"),
        abs_med=("abs_v","median"),
        n=("sample_id","size")
    )
    pri=d.assign(_cls=cls.astype(int)).groupby("speaker_id")._cls.mean()
    pri=pri.reindex(sp.index).fillna(0)
    bpc=safe_corr(sp.baseline.to_numpy(),pri.to_numpy())
    between=float(np.std(sp.baseline.to_numpy(),ddof=0))
    within=float(d.groupby("speaker_id").rel_v.std().fillna(0).mean())
    ratio=float(between/max(within,1e-8))
    corpus_ref=float(np.median(sp.abs_med.to_numpy()))
    abs_dev=d.abs_v.to_numpy(float)-corpus_ref
    rel=d.rel_v.to_numpy(float)
    conflict=(np.sign(abs_dev)*np.sign(rel)<0)&(np.abs(abs_dev)>=a["cabs"])&(np.abs(rel)>=a["crel"])
    cf=float(conflict.mean())
    return {
      "dataset":dataset,"attribute":attr,"emotion":emotion,
      "class_prevalence":prev,"speaker_coverage":sp_cov,
      "absolute_effect_signed":ae,"relative_effect_signed":re,"baseline_effect_signed":be,
      "absolute_effect_abs":abs(ae),"relative_effect_abs":abs(re),"baseline_effect_abs":abs(be),
      "relative_minus_absolute_separation":abs(re)-abs(ae),
      "baseline_prior_corr":bpc,
      "between_baseline_to_within_relative_sd_ratio":ratio,
      "conflict_fraction":cf
    }

def load_targets():
    p9=pd.read_csv("results/EXP-20260920-09/per_emotion_deltas.csv")
    p10=pd.read_csv("results/EXP-20260920-10/per_emotion_deltas.csv")
    p9=p9[p9.comparison.eq("relative-absolute")].copy()
    p9["attribute"]="pitch"
    p10=p10[p10.comparison.eq("relative-absolute")].copy()
    x=pd.concat([
      p9[["dataset","attribute","emotion","delta_mean","ci95_low","ci95_high"]],
      p10[["dataset","attribute","emotion","delta_mean","ci95_low","ci95_high"]]
    ],ignore_index=True)
    x["target_positive"]=(x.delta_mean>0).astype(int)
    x["resolved"]=((x.ci95_low>0)|(x.ci95_high<0))
    return x

def make_pipe(kind,family,alpha,C):
    if family=="labels_only":
        pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),CAT)],remainder="drop")
    elif family=="stats_only":
        pre=ColumnTransformer([("num",StandardScaler(),NUMERIC)],remainder="drop")
    elif family=="combined":
        pre=ColumnTransformer([
          ("num",StandardScaler(),NUMERIC),
          ("cat",OneHotEncoder(handle_unknown="ignore"),CAT)
        ],remainder="drop")
    else:
        raise ValueError(family)
    if kind=="reg":
        est=Ridge(alpha=float(alpha))
    else:
        est=LogisticRegression(C=float(C),class_weight="balanced",max_iter=5000,solver="lbfgs")
    return Pipeline([("pre",pre),("est",est)])

def fold_predictions(df,hold_col,alpha,C):
    rows=[]
    for hold in sorted(df[hold_col].unique()):
        tr=df[df[hold_col].ne(hold)].copy()
        te=df[df[hold_col].eq(hold)].copy()
        for fam in ["labels_only","stats_only","combined"]:
            reg=make_pipe("reg",fam,alpha,C)
            clf=make_pipe("clf",fam,alpha,C)
            reg.fit(tr, tr.delta_mean)
            clf.fit(tr, tr.target_positive)
            pred=reg.predict(te)
            prob=clf.predict_proba(te)[:,list(clf.named_steps["est"].classes_).index(1)]
            sign=(prob>=.5).astype(int)
            for j,(_,r) in enumerate(te.iterrows()):
                rows.append({
                  "split_type":hold_col,"held_out":hold,"model_family":fam,
                  "dataset":r.dataset,"attribute":r.attribute,"emotion":r.emotion,
                  "target_delta":r.delta_mean,"target_positive":int(r.target_positive),
                  "resolved":bool(r.resolved),"predicted_delta":float(pred[j]),
                  "predicted_positive":int(sign[j]),"positive_probability":float(prob[j])
                })
    return pd.DataFrame(rows)

def metrics(pred):
    out=[]
    for (split,fam),g in pred.groupby(["split_type","model_family"]):
        rho=float(spearmanr(g.target_delta,g.predicted_delta).statistic)
        mae=float(mean_absolute_error(g.target_delta,g.predicted_delta))
        acc=float(accuracy_score(g.target_positive,g.predicted_positive))
        gr=g[g.resolved]
        racc=float(accuracy_score(gr.target_positive,gr.predicted_positive))
        rbal=float(balanced_accuracy_score(gr.target_positive,gr.predicted_positive))
        out.append({"split_type":split,"model_family":fam,"n":len(g),"n_resolved":len(gr),
                    "mae":mae,"spearman_r":rho,"accuracy_all":acc,
                    "accuracy_resolved":racc,"balanced_accuracy_resolved":rbal})
    return pd.DataFrame(out)

def heldout_metrics(pred,hold_col):
    rows=[]
    for (hold,fam),g in pred.groupby(["held_out","model_family"]):
        gr=g[g.resolved]
        rows.append({
          "held_out":hold,"model_family":fam,"n":len(g),"n_resolved":len(gr),
          "mae":float(mean_absolute_error(g.target_delta,g.predicted_delta)),
          "spearman_r":float(spearmanr(g.target_delta,g.predicted_delta).statistic),
          "accuracy_all":float(accuracy_score(g.target_positive,g.predicted_positive)),
          "accuracy_resolved":float(accuracy_score(gr.target_positive,gr.predicted_positive)) if len(gr) else np.nan,
          "balanced_accuracy_resolved":float(balanced_accuracy_score(gr.target_positive,gr.predicted_positive)) if len(gr) else np.nan
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(src,columns=COLS)

    feats=[]
    for ds in cfg["dataset"]["corpora"]:
      for attr in cfg["dataset"]["attributes"]:
        d=prepare(raw,ds,attr)
        for emo in cfg["dataset"]["emotions"]:
          feats.append(feature_row(d,ds,attr,emo))
    f=pd.DataFrame(feats)
    t=load_targets()
    x=f.merge(t,on=["dataset","attribute","emotion"],how="inner",validate="one_to_one")
    if len(x)!=75 or x[["dataset","attribute","emotion"]].duplicated().any():
        raise RuntimeError(f"expected 75 unique cells, got {len(x)}")

    loco=fold_predictions(x,"dataset",cfg["parameters"]["ridge_alpha"],cfg["parameters"]["logistic_C"])
    loao=fold_predictions(x,"attribute",cfg["parameters"]["ridge_alpha"],cfg["parameters"]["logistic_C"])
    lm=metrics(loco); am=metrics(loao)
    lh=heldout_metrics(loco,"dataset"); ah=heldout_metrics(loao,"attribute")

    x.to_csv(root/"effect_matrix.csv",index=False)
    f.to_csv(root/"predictor_features.csv",index=False)
    loco.to_csv(root/"loco_predictions.csv",index=False)
    pd.concat([lm.assign(level="pooled"),lh.assign(split_type="dataset",level="held_out")],ignore_index=True,sort=False).to_csv(root/"loco_metrics.csv",index=False)
    loao.to_csv(root/"loao_predictions.csv",index=False)
    pd.concat([am.assign(level="pooled"),ah.assign(split_type="attribute",level="held_out")],ignore_index=True,sort=False).to_csv(root/"loao_metrics.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"source":str(src),"cells":len(x),
      "resolved_cells":int(x.resolved.sum()),"numeric_features":NUMERIC,"categorical_features":CAT,
      "ridge_alpha":cfg["parameters"]["ridge_alpha"],"logistic_C":cfg["parameters"]["logistic_C"]
    },indent=2)+"\n")
    print("LOCO pooled")
    print(lm.to_string(index=False))
    print("\nLOCO held out")
    print(lh.to_string(index=False))
    print("\nLOAO pooled")
    print(am.to_string(index=False))
    print("\nLOAO held out")
    print(ah.to_string(index=False))

if __name__=="__main__":
    main()
