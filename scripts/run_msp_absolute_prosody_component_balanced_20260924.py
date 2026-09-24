#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from run_iemocap_component_balanced_ensemble_20260924 import (
    precompute_component_arrays, optimize_mean, optimize_maximin
)
from run_iemocap_nested_component_recomposition_20260924 import (
    equal_speaker_weights, speaker_moments, ccc_from_mom, simplex_weights
)

FEATURES = {
    "pitch_level": ["pitch_abs"],
    "pitch_shape": ["pitch_abs", "f0var_abs"],
    "loudness": ["loud_abs"],
    "temporal": ["rate_abs", "pause_abs", "voiced_abs"],
    "full_linear": ["pitch_abs", "f0var_abs", "loud_abs", "rate_abs", "pause_abs", "voiced_abs"],
    "full_quadratic": ["pitch_abs", "f0var_abs", "loud_abs", "rate_abs", "pause_abs", "voiced_abs"],
}
CANDIDATES = list(FEATURES)
METHODS = ["nested_best_single", "equal_ensemble", "inner_overall_convex", "component_balanced_mean", "component_balanced_maximin"]

def stable_seed(*parts):
    h=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],"big")%(2**32)

def fold_map(speakers, nfolds, seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    np.random.default_rng(int(seed)).shuffle(s)
    return {sp:i % int(nfolds) for i,sp in enumerate(s)}

def prepare(path):
    cols=[
        "sample_id","dataset","speaker_id","arousal_mean_1_7","dominance_mean_1_7",
        "f0_median_hz","f0_iqr_semitone","speech_lufs","phoneme_articulation_rate",
        "pause_ratio","voiced_ratio","pitch_reference_scope","loudness_reference_scope","rate_reference_scope"
    ]
    d=pd.read_parquet(path,columns=cols)
    d=d[d.dataset.eq("msp")].dropna().copy()
    d=d[
        (d.speaker_id.astype(str)!="Unknown") &
        (d.f0_median_hz>0) &
        (d.phoneme_articulation_rate>0) &
        d.pitch_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"]) &
        d.loudness_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"]) &
        d.rate_reference_scope.isin(["speaker_neutral","speaker_neutral_shrunk"])
    ].copy()
    n=d.groupby("speaker_id").size()
    d=d[d.speaker_id.isin(n[n>10].index)].copy()
    d["speaker_id"]=d.speaker_id.astype(str)
    d["pitch_abs"]=12*np.log2(d.f0_median_hz.astype(float))
    d["f0var_abs"]=d.f0_iqr_semitone.astype(float)
    d["loud_abs"]=d.speech_lufs.astype(float)
    d["rate_abs"]=np.log(d.phoneme_articulation_rate.astype(float))
    d["pause_abs"]=d.pause_ratio.astype(float)
    d["voiced_abs"]=d.voiced_ratio.astype(float)
    d=d.rename(columns={"arousal_mean_1_7":"arousal","dominance_mean_1_7":"dominance"})
    d=d.sort_values("sample_id").reset_index(drop=True)
    if d.sample_id.duplicated().any():
        raise RuntimeError("duplicate sample_id")
    if len(d)!=197014 or d.speaker_id.nunique()!=1915:
        raise RuntimeError(f"unexpected filtered inventory rows={len(d)} speakers={d.speaker_id.nunique()}")
    return d

def fit_predict(train, test, cols, Ytr, w, alpha, solver, quadratic):
    Xtr=train[cols].to_numpy(np.float32)
    Xte=test[cols].to_numpy(np.float32)
    sc=StandardScaler()
    sc.fit(Xtr,sample_weight=w)
    A=sc.transform(Xtr); B=sc.transform(Xte)
    if quadratic:
        poly=PolynomialFeatures(degree=2,include_bias=False)
        A=poly.fit_transform(A); B=poly.transform(B)
        sc2=StandardScaler()
        sc2.fit(A,sample_weight=w)
        A=sc2.transform(A); B=sc2.transform(B)
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always")
        model=Ridge(alpha=float(alpha),solver=str(solver))
        model.fit(A,Ytr,sample_weight=w)
        pred=model.predict(B)
    return pred, len(ww)

def comp_scores(df):
    sp,mm=speaker_moments(df)
    return {m:float(ccc_from_mom(v.mean(0))) for m,v in mm.items()}

def make_eval_df(q, idx, y, pred):
    return pd.DataFrame({
        "sample_id":q.loc[idx,"sample_id"].to_numpy(),
        "speaker_id":q.loc[idx,"speaker_id"].to_numpy(),
        "y_true":np.asarray(y,float),
        "pred":np.asarray(pred,float),
    })

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)

    q=prepare(cfg["source_parquet"])
    Y=q[["arousal","dominance"]].to_numpy(float)
    speakers=sorted(q.speaker_id.unique())
    seeds=[int(x) for x in cfg["seeds"]]
    outer_folds=int(cfg["outer_folds"]); inner_folds=int(cfg["inner_folds"])
    alpha=float(cfg["ridge_alpha"]); solver=str(cfg["ridge_solver"])

    out_chunks=[]; selection_rows=[]; fit_rows=[]
    for seed in seeds:
        fmap=fold_map(speakers,outer_folds,seed)
        outer_assign=q.speaker_id.map(fmap).to_numpy(int)
        for outer in range(outer_folds):
            trmask=outer_assign!=outer
            temask=~trmask
            train_idx=np.where(trmask)[0]
            test_idx=np.where(temask)[0]
            train_speakers=sorted(q.loc[train_idx,"speaker_id"].unique())
            imap=fold_map(train_speakers,inner_folds,stable_seed(seed,"inner",outer))
            inner_assign=q.loc[train_idx,"speaker_id"].map(imap).to_numpy(int)

            inner_pred={n:np.full((len(train_idx),2),np.nan,float) for n in CANDIDATES}
            outer_pred={}
            for inner in range(inner_folds):
                iv_local=np.where(inner_assign==inner)[0]
                it_local=np.where(inner_assign!=inner)[0]
                fit_idx=train_idx[it_local]; val_idx=train_idx[iv_local]
                w=equal_speaker_weights(q.loc[fit_idx,"speaker_id"].to_numpy())
                for name in CANDIDATES:
                    quad=name=="full_quadratic"
                    pred,nwarn=fit_predict(q.loc[fit_idx],q.loc[val_idx],FEATURES[name],Y[fit_idx],w,alpha,solver,quad)
                    inner_pred[name][iv_local]=pred
                    fit_rows.append({
                        "seed":seed,"outer_fold":outer,"inner_fold":inner,"stage":"inner",
                        "candidate":name,"train_rows":len(fit_idx),"test_rows":len(val_idx),
                        "train_speakers":q.loc[fit_idx,"speaker_id"].nunique(),
                        "test_speakers":q.loc[val_idx,"speaker_id"].nunique(),"warning_count":nwarn
                    })

            wouter=equal_speaker_weights(q.loc[train_idx,"speaker_id"].to_numpy())
            for name in CANDIDATES:
                quad=name=="full_quadratic"
                pred,nwarn=fit_predict(q.loc[train_idx],q.loc[test_idx],FEATURES[name],Y[train_idx],wouter,alpha,solver,quad)
                outer_pred[name]=pred
                fit_rows.append({
                    "seed":seed,"outer_fold":outer,"inner_fold":-1,"stage":"outer",
                    "candidate":name,"train_rows":len(train_idx),"test_rows":len(test_idx),
                    "train_speakers":q.loc[train_idx,"speaker_id"].nunique(),
                    "test_speakers":q.loc[test_idx,"speaker_id"].nunique(),"warning_count":nwarn
                })

            if any(np.isnan(inner_pred[n]).any() for n in CANDIDATES):
                raise RuntimeError(f"missing inner predictions seed={seed} outer={outer}")

            for j,target in enumerate(["arousal","dominance"]):
                scores={}
                for name in CANDIDATES:
                    z=make_eval_df(q,train_idx,Y[train_idx,j],inner_pred[name][:,j])
                    scores[name]=comp_scores(z)
                best=max(CANDIDATES,key=lambda n:scores[n]["overall"])
                base_scores=np.array([scores[best]["overall"],scores[best]["between"],scores[best]["within"]],float)

                Pinner=np.column_stack([inner_pred[n][:,j] for n in CANDIDATES])
                pre=precompute_component_arrays(Y[train_idx,j],Pinner,q.loc[train_idx,"speaker_id"])
                w_equal=np.full(len(CANDIDATES),1/len(CANDIDATES))
                w_conv,ok_conv,msg_conv=simplex_weights(Y[train_idx,j],Pinner,q.loc[train_idx,"speaker_id"])
                one=np.zeros(len(CANDIDATES)); one[CANDIDATES.index(best)]=1.0
                w_mean,ok_mean,msg_mean,mean_scores=optimize_mean(pre,[w_equal,w_conv,one,*np.eye(len(CANDIDATES))])
                w_max,ok_max,msg_max,max_scores,tmax=optimize_maximin(pre,base_scores,[w_equal,w_conv,w_mean,one,*np.eye(len(CANDIDATES))])

                Pouter=np.column_stack([outer_pred[n][:,j] for n in CANDIDATES])
                pred_map={
                    "nested_best_single":outer_pred[best][:,j],
                    "equal_ensemble":Pouter@w_equal,
                    "inner_overall_convex":Pouter@w_conv,
                    "component_balanced_mean":Pouter@w_mean,
                    "component_balanced_maximin":Pouter@w_max,
                }
                for method,pred in pred_map.items():
                    out_chunks.append(pd.DataFrame({
                        "sample_id":q.loc[test_idx,"sample_id"].to_numpy(),
                        "speaker_id":q.loc[test_idx,"speaker_id"].to_numpy(),
                        "seed":seed,"outer_fold":outer,"target":target,"method":method,
                        "y_true":Y[test_idx,j],"pred":pred
                    }))
                selection_rows.append({
                    "seed":seed,"outer_fold":outer,"target":target,"best_single":best,
                    "best_inner_overall":base_scores[0],"best_inner_between":base_scores[1],"best_inner_within":base_scores[2],
                    "convex_ok":bool(ok_conv),"convex_message":str(msg_conv),
                    "convex_weights":json.dumps({n:float(w) for n,w in zip(CANDIDATES,w_conv)},sort_keys=True),
                    "mean_ok":bool(ok_mean),"mean_message":str(msg_mean),
                    "mean_weights":json.dumps({n:float(w) for n,w in zip(CANDIDATES,w_mean)},sort_keys=True),
                    "mean_inner_overall":float(mean_scores[0]),"mean_inner_between":float(mean_scores[1]),"mean_inner_within":float(mean_scores[2]),
                    "maximin_ok":bool(ok_max),"maximin_message":str(msg_max),
                    "maximin_weights":json.dumps({n:float(w) for n,w in zip(CANDIDATES,w_max)},sort_keys=True),
                    "maximin_inner_t":float(tmax),"maximin_inner_overall":float(max_scores[0]),
                    "maximin_inner_between":float(max_scores[1]),"maximin_inner_within":float(max_scores[2]),
                })
                print("SELECT",seed,outer,target,"best",best,"t",round(float(tmax),6),flush=True)

    O=pd.concat(out_chunks,ignore_index=True)
    O.to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(selection_rows).to_csv(root/"selection_audit.csv",index=False)
    pd.DataFrame(fit_rows).to_csv(root/"training_audit.csv",index=False)

    metric_rows=[]; per_seed_delta=[]; moments_cache={}
    for seed in seeds:
        for target in ["arousal","dominance"]:
            zt=O[(O.seed==seed)&(O.target==target)]
            for method in METHODS:
                sp,mm=speaker_moments(zt[zt.method==method])
                if sp!=speakers:
                    raise RuntimeError(f"speaker alignment fail {seed}/{target}/{method}")
                moments_cache[(seed,target,method)]=mm
                for metric in ["overall","between","within"]:
                    metric_rows.append({
                        "seed":seed,"target":target,"method":method,"metric":metric,
                        "ccc":float(ccc_from_mom(mm[metric].mean(0)))
                    })
            for method in METHODS:
                if method=="nested_best_single": continue
                for metric in ["overall","between","within"]:
                    a=moments_cache[(seed,target,method)][metric]
                    b=moments_cache[(seed,target,"nested_best_single")][metric]
                    per_seed_delta.append({
                        "seed":seed,"target":target,"method":method,"reference":"nested_best_single","metric":metric,
                        "delta_ccc":float(ccc_from_mom(a.mean(0))-ccc_from_mom(b.mean(0)))
                    })

    pd.DataFrame(metric_rows).to_csv(root/"component_metrics_by_seed.csv",index=False)
    S=pd.DataFrame(per_seed_delta)
    S.to_csv(root/"deltas_by_seed.csv",index=False)

    reps=int(cfg["bootstrap_reps"]); bs=int(cfg["bootstrap_seed"]); ns=len(speakers)
    summary=[]
    comparisons=[
        ("equal_ensemble","nested_best_single"),
        ("inner_overall_convex","nested_best_single"),
        ("component_balanced_mean","nested_best_single"),
        ("component_balanced_maximin","nested_best_single"),
        ("component_balanced_maximin","inner_overall_convex"),
    ]
    for target in ["arousal","dominance"]:
        for method,ref in comparisons:
            for metric in ["overall","between","within"]:
                rng=np.random.default_rng(stable_seed(bs,target,method,ref,metric))
                counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
                boot=[]; pts=[]
                for seed in seeds:
                    A=moments_cache[(seed,target,method)][metric]
                    B=moments_cache[(seed,target,ref)][metric]
                    va=ccc_from_mom((counts@A)/ns)
                    vb=ccc_from_mom((counts@B)/ns)
                    boot.append(va-vb)
                    pts.append(float(ccc_from_mom(A.mean(0))-ccc_from_mom(B.mean(0))))
                bb=np.mean(np.vstack(boot),axis=0)
                lo,hi=np.nanquantile(bb,[.025,.975])
                summary.append({
                    "target":target,"method":method,"reference":ref,"metric":metric,
                    "delta_ccc_mean_across_seeds":float(np.mean(pts)),
                    "ci95_low":float(lo),"ci95_high":float(hi),
                    "ci_excludes_zero":bool(lo>0 or hi<0),
                    "positive_seeds":int(sum(x>0 for x in pts)),"n_seeds":len(seeds),
                })
    D=pd.DataFrame(summary)
    D.to_csv(root/"paired_component_deltas.csv",index=False)

    prim=D[(D.method=="component_balanced_maximin")&(D.reference=="nested_best_single")]
    checks={}
    for target in ["arousal","dominance"]:
        g=prim[prim.target==target]
        checks[target+"_all_three_mean_positive"]=bool((g.delta_ccc_mean_across_seeds>0).all() and len(g)==3)
        checks[target+"_all_three_seed_stable"]=bool((g.positive_seeds>=2).all() and len(g)==3)
    decision="pass" if all(checks.values()) else ("mixed" if any(checks.values()) else "reject")
    audit={
        "experiment_id":cfg["experiment_id"],"validity":"valid","decision":decision,
        "task":"standard absolute MSP Arousal/Dominance",
        "rows":len(q),"speakers":len(speakers),"seeds":seeds,
        "outer_split":"5-fold speaker-disjoint","inner_split":"4-fold nested speaker-disjoint",
        "relative_role":"diagnostic decomposition only; all candidate inputs are absolute prosody",
        "candidates":CANDIDATES,
        "fit_warning_count_total":int(pd.DataFrame(fit_rows).warning_count.sum()),
        "checks":checks,
        "selection_leakage_check":"pass by construction: every outer test speaker is excluded from all inner fitting and selection",
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_parquet":cfg["source_parquet"],"ridge_alpha":alpha,"ridge_solver":solver,
        "bootstrap_unit":"speaker","bootstrap_reps":reps,"bootstrap_seed":bs
    },indent=2)+"\n")
    print("\nPRIMARY")
    print(prim.to_string(index=False))
    print("\nAUDIT")
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
