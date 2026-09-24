#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_iemocap_nested_component_recomposition_20260924 import (
    speaker_id, session_id, load_simple, load_wavlm, align_matrix, fit_fold,
    speaker_moments, ccc_from_mom, make_df, simplex_weights
)
from run_iemocap_component_balanced_ensemble_20260924 import (
    precompute_component_arrays, comp_scores_fast, optimize_mean, optimize_maximin, SSL
)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)

    cols=["file","EmoAct","EmoDom","speaking_rate","pitch_mean"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(cfg["dataset_glob"]))],
                ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str)
    d["speaker_id"]=d.file.map(speaker_id)
    d["session_id"]=d.file.map(session_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})

    emb=load_wavlm(cfg["embeddings"]["wavlm"])
    for name in ["emotion2vec_plus_large","hubert_base_ls960","wav2vec2_base"]:
        emb[name]=load_simple(cfg["embeddings"][name])
    common=set(d.sample_id.astype(str))
    for tab,_ in emb.values(): common &= set(tab.sample_id.astype(str))
    q=d[d.sample_id.astype(str).isin(common)].sort_values("sample_id").reset_index(drop=True)
    if len(q)!=10039: raise RuntimeError(len(q))
    Y=q[["arousal","dominance"]].to_numpy(float)
    models={name:(align_matrix(q,tab,arr),False) for name,(tab,arr) in emb.items()}

    train_sessions=["Ses01","Ses02","Ses03"]; val_session="Ses04"; test_session="Ses05"
    tr=q.session_id.isin(train_sessions).to_numpy()
    va=(q.session_id.to_numpy()==val_session)
    te=(q.session_id.to_numpy()==test_session)
    predict_mask=va|te
    pred={}; fit_rows=[]
    for name in SSL:
        X,quad=models[name]
        pp,ctx,nwarn=fit_fold(X,Y,q,tr,predict_mask,float(cfg["ridge_alpha"]),str(cfg["ridge_solver"]),quad)
        full=np.full((len(q),2),np.nan,float); full[predict_mask]=pp
        pred[name]=full
        fit_rows.append({"model":name,"train_rows":int(tr.sum()),"validation_rows":int(va.sum()),"test_rows":int(te.sum()),
                         "train_speakers":int(q.loc[tr,"speaker_id"].nunique()),
                         "validation_speakers":int(q.loc[va,"speaker_id"].nunique()),
                         "test_speakers":int(q.loc[te,"speaker_id"].nunique()),"warning_count":int(nwarn)})
        print("FIT",name,"warnings",nwarn,flush=True)

    val_idx=np.where(va)[0]; test_idx=np.where(te)[0]
    methods=["best_single","equal_ssl_ensemble","validation_convex_ssl_ensemble",
             "component_balanced_mean","component_balanced_maximin"]
    rows=[]; selection=[]
    for j,target in enumerate(["arousal","dominance"]):
        scores={}
        for name in SSL:
            z=make_df(q,Y,j,val_idx,pred[name][val_idx,j])
            sp,mm=speaker_moments(z)
            scores[name]={k:float(ccc_from_mom(v.mean(0))) for k,v in mm.items()}
        best=max(SSL,key=lambda n:scores[n]["overall"])
        base_scores=np.array([scores[best]["overall"],scores[best]["between"],scores[best]["within"]],float)

        Pval=np.column_stack([pred[n][val_idx,j] for n in SSL])
        pre=precompute_component_arrays(Y[val_idx,j],Pval,q.loc[val_idx,"speaker_id"])
        w_equal=np.full(len(SSL),1/len(SSL))
        w_conv,ok_conv,msg_conv=simplex_weights(Y[val_idx,j],Pval,q.loc[val_idx,"speaker_id"])
        one=np.zeros(len(SSL)); one[SSL.index(best)]=1.0
        w_mean,ok_mean,msg_mean,mean_scores=optimize_mean(pre,[w_equal,w_conv,one,*np.eye(len(SSL))])
        w_max,ok_max,msg_max,max_scores,tmax=optimize_maximin(pre,base_scores,[w_equal,w_conv,w_mean,one,*np.eye(len(SSL))])

        Ptest=np.column_stack([pred[n][test_idx,j] for n in SSL])
        pred_map={
            "best_single":pred[best][test_idx,j],
            "equal_ssl_ensemble":Ptest@w_equal,
            "validation_convex_ssl_ensemble":Ptest@w_conv,
            "component_balanced_mean":Ptest@w_mean,
            "component_balanced_maximin":Ptest@w_max,
        }
        for m,pv in pred_map.items():
            for ii,x in zip(test_idx,pv):
                rows.append({"sample_id":q.loc[ii,"sample_id"],"speaker_id":q.loc[ii,"speaker_id"],
                             "session_id":test_session,"target":target,"method":m,
                             "y_true":float(Y[ii,j]),"pred":float(x)})
        selection.append({
            "target":target,"best_single":best,
            "best_val_overall":float(base_scores[0]),"best_val_between":float(base_scores[1]),"best_val_within":float(base_scores[2]),
            "convex_ok":bool(ok_conv),"convex_message":msg_conv,
            "convex_weights":json.dumps({n:float(w) for n,w in zip(SSL,w_conv)},sort_keys=True),
            "mean_ok":bool(ok_mean),"mean_message":msg_mean,
            "mean_weights":json.dumps({n:float(w) for n,w in zip(SSL,w_mean)},sort_keys=True),
            "mean_val_overall":float(mean_scores[0]),"mean_val_between":float(mean_scores[1]),"mean_val_within":float(mean_scores[2]),
            "maximin_ok":bool(ok_max),"maximin_message":msg_max,
            "maximin_weights":json.dumps({n:float(w) for n,w in zip(SSL,w_max)},sort_keys=True),
            "maximin_val_t":float(tmax),"maximin_val_overall":float(max_scores[0]),
            "maximin_val_between":float(max_scores[1]),"maximin_val_within":float(max_scores[2]),
        })

    O=pd.DataFrame(rows)
    O.to_parquet(root/"test_predictions.parquet",index=False)
    pd.DataFrame(fit_rows).to_csv(root/"training_audit.csv",index=False)
    pd.DataFrame(selection).to_csv(root/"selection_audit.csv",index=False)

    metric_rows=[]; ref_rows=[]
    refs=cfg["published_pcm"]
    for target in ["arousal","dominance"]:
        zt=O[O.target==target]
        for m in methods:
            sp,mm=speaker_moments(zt[zt.method==m])
            for met in ["overall","between","within"]:
                metric_rows.append({"target":target,"method":m,"metric":met,
                                    "ccc":float(ccc_from_mom(mm[met].mean(0)))})
        pub=float(refs[target])
        for m in methods:
            obs=float([r["ccc"] for r in metric_rows if r["target"]==target and r["method"]==m and r["metric"]=="overall"][0])
            ref_rows.append({"target":target,"method":m,"our_overall_ccc":obs,
                             "published_pcm_ccc":pub,"delta_vs_published_pcm":obs-pub,
                             "published_doi":refs["doi"]})
    M=pd.DataFrame(metric_rows); R=pd.DataFrame(ref_rows)
    M.to_csv(root/"component_metrics.csv",index=False)
    R.to_csv(root/"published_pcm_comparison.csv",index=False)

    # Test deltas vs validation-selected best single.
    D=[]
    for target in ["arousal","dominance"]:
        z=M[M.target==target].pivot(index="method",columns="metric",values="ccc")
        for m in methods:
            if m=="best_single": continue
            for met in ["overall","between","within"]:
                D.append({"target":target,"method":m,"reference":"best_single","metric":met,
                          "delta_ccc":float(z.loc[m,met]-z.loc["best_single",met])})
    pd.DataFrame(D).to_csv(root/"deltas_vs_best_single.csv",index=False)

    p=pd.DataFrame(D); prim=p[p.method=="component_balanced_maximin"]
    positive=[]
    for t,g in prim.groupby("target"):
        dd={r.metric:r.delta_ccc for r in g.itertuples()}
        if all(dd[k]>0 for k in ["overall","between","within"]): positive.append(t)
    audit={
        "experiment_id":cfg["experiment_id"],"validity":"valid",
        "decision":"pass" if positive else "mixed",
        "split":{"train":train_sessions,"validation":[val_session],"test":[test_session]},
        "test_rows":int(te.sum()),"test_speakers":int(q.loc[te,"speaker_id"].nunique()),
        "positive_all_three_targets_for_maximin":positive,
        "published_pcm":{"arousal":float(refs["arousal"]),"dominance":float(refs["dominance"]),"doi":refs["doi"]},
        "comparability_note":"Same IEMOCAP session split and CCC dimensions as PCM; architecture, optimization, and training regime differ.",
        "fit_warning_count_total":int(pd.DataFrame(fit_rows).warning_count.sum())
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "task":"raw absolute IEMOCAP Arousal/Dominance","split":"Ses01-03 train / Ses04 validation / Ses05 test"
    },indent=2)+"\n")
    print(M.to_string(index=False))
    print("\nPCM COMPARISON")
    print(R.to_string(index=False))
    print("\nAUDIT",json.dumps(audit,indent=2))

if __name__=="__main__": main()
