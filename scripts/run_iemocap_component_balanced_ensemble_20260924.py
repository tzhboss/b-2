#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, itertools, json, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from run_iemocap_nested_component_recomposition_20260924 import (
    speaker_id, session_id, load_simple, load_wavlm, align_matrix, fit_fold,
    split_enrollment, equal_speaker_weights, weighted_ccc, speaker_moments,
    ccc_from_mom, make_df, simplex_weights
)

SSL=["wavlm_layer12","wavlm_layer24","emotion2vec_plus_large","hubert_base_ls960","wav2vec2_base"]

def precompute_component_arrays(y,P,speakers):
    y=np.asarray(y,float); P=np.asarray(P,float); s=np.asarray(speakers).astype(str)
    obs_w=equal_speaker_weights(s)
    uniq=sorted(np.unique(s))
    smap={sp:i for i,sp in enumerate(uniq)}
    gi=np.array([smap[x] for x in s],int)
    ym=np.array([y[gi==i].mean() for i in range(len(uniq))],float)
    Pm=np.vstack([P[gi==i].mean(0) for i in range(len(uniq))])
    yc=y-ym[gi]
    Pc=P-Pm[gi]
    return dict(y=y,P=P,obs_w=obs_w,ym=ym,Pm=Pm,yc=yc,Pc=Pc)

def comp_scores_fast(pre,w):
    w=np.asarray(w,float)
    return np.array([
        weighted_ccc(pre["y"],pre["P"]@w,pre["obs_w"]),
        weighted_ccc(pre["ym"],pre["Pm"]@w,np.ones(len(pre["ym"]))),
        weighted_ccc(pre["yc"],pre["Pc"]@w,pre["obs_w"]),
    ],float)

def optimize_mean(pre,starts):
    m=pre["P"].shape[1]
    cons={"type":"eq","fun":lambda w:np.sum(w)-1.0}
    best=None
    for x0 in starts:
        res=minimize(lambda w:-float(np.mean(comp_scores_fast(pre,w))),np.asarray(x0,float),
                     method="SLSQP",bounds=[(0,1)]*m,constraints=[cons],
                     options={"maxiter":500,"ftol":1e-10})
        w=np.maximum(res.x,0); w=w/max(w.sum(),1e-12)
        sc=comp_scores_fast(pre,w)
        key=(float(np.mean(sc)),float(sc[0]))
        if best is None or key>best[0]: best=(key,w,bool(res.success),str(res.message),sc)
    return best[1],best[2],best[3],best[4]

def optimize_maximin(pre,base_scores,starts):
    m=pre["P"].shape[1]
    best=None
    for w0 in starts:
        w0=np.asarray(w0,float); w0=np.maximum(w0,0); w0=w0/max(w0.sum(),1e-12)
        sc0=comp_scores_fast(pre,w0)
        t0=min(sc0[1]-base_scores[1],sc0[2]-base_scores[2])
        x0=np.r_[w0,t0]
        def scores(x): return comp_scores_fast(pre,x[:m])
        cons=[
            {"type":"eq","fun":lambda x:np.sum(x[:m])-1.0},
            {"type":"ineq","fun":lambda x:scores(x)[1]-base_scores[1]-x[m]},
            {"type":"ineq","fun":lambda x:scores(x)[2]-base_scores[2]-x[m]},
            {"type":"ineq","fun":lambda x:scores(x)[0]-base_scores[0]},
        ]
        res=minimize(lambda x:-x[m]-1e-3*scores(x)[0],x0,method="SLSQP",
                     bounds=[(0,1)]*m+[(-1,1)],constraints=cons,
                     options={"maxiter":1000,"ftol":1e-11})
        w=np.maximum(res.x[:m],0); w=w/max(w.sum(),1e-12)
        sc=comp_scores_fast(pre,w)
        t=min(sc[1]-base_scores[1],sc[2]-base_scores[2])
        feas=(sc[0]>=base_scores[0]-1e-6)
        key=(bool(feas),float(t),float(sc[0]))
        if best is None or key>best[0]: best=(key,w,bool(res.success),str(res.message),sc,t)
    return best[1],best[2],best[3],best[4],best[5]

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
    sessions=sorted(q.session_id.unique())
    Y=q[["arousal","dominance"]].to_numpy(float)
    models={name:(align_matrix(q,tab,arr),False) for name,(tab,arr) in emb.items()}

    pred_cache={}; fit_rows=[]
    held_sets=[(s,) for s in sessions]+list(itertools.combinations(sessions,2))
    for held in held_sets:
        tr=~q.session_id.isin(held).to_numpy(); te=~tr
        for name in SSL:
            X,quad=models[name]
            pred,ctx,nwarn=fit_fold(X,Y,q,tr,te,float(cfg["ridge_alpha"]),str(cfg["ridge_solver"]),quad)
            full=np.full((len(q),2),np.nan,float); full[te]=pred
            pred_cache[(tuple(held),name)]=full
            fit_rows.append({"held_sessions":"+".join(held),"model":name,"train_rows":int(tr.sum()),
                             "test_rows":int(te.sum()),"train_speakers":int(q.loc[tr,"speaker_id"].nunique()),
                             "test_speakers":int(q.loc[te,"speaker_id"].nunique()),"warning_count":int(nwarn)})
            print("FIT",held,name,"warnings",nwarn,flush=True)

    methods=["nested_best_single","equal_ssl_ensemble","inner_convex_ssl_ensemble",
             "component_balanced_mean","component_balanced_maximin"]
    out=[]; sel=[]
    K=int(cfg["evaluation_enrollment_k"])

    for outer in sessions:
        train_sessions=[s for s in sessions if s!=outer]
        inner_idx=np.where(q.session_id.isin(train_sessions).to_numpy())[0]
        inner_pred={n:np.full((len(q),2),np.nan,float) for n in SSL}
        for inner in train_sessions:
            keyheld=tuple(sorted((outer,inner)))
            vi=np.where(q.session_id.to_numpy()==inner)[0]
            for n in SSL: inner_pred[n][vi]=pred_cache[(keyheld,n)][vi]
        outer_idx=np.where(q.session_id.to_numpy()==outer)[0]
        for j,target in enumerate(["arousal","dominance"]):
            # Model-level inner component scores.
            scores={}
            for n in SSL:
                z=make_df(q,Y,j,inner_idx,inner_pred[n][inner_idx,j])
                sp,mm=speaker_moments(z)
                scores[n]={k:float(ccc_from_mom(v.mean(0))) for k,v in mm.items()}
            best=max(SSL,key=lambda n:scores[n]["overall"])
            base_scores=np.array([scores[best]["overall"],scores[best]["between"],scores[best]["within"]],float)

            Pinner=np.column_stack([inner_pred[n][inner_idx,j] for n in SSL])
            pre=precompute_component_arrays(Y[inner_idx,j],Pinner,q.loc[inner_idx,"speaker_id"])
            w_equal=np.full(len(SSL),1/len(SSL))
            w_conv,ok_conv,msg_conv=simplex_weights(Y[inner_idx,j],Pinner,q.loc[inner_idx,"speaker_id"])
            one=np.zeros(len(SSL)); one[SSL.index(best)]=1.0
            w_mean,ok_mean,msg_mean,inner_mean_scores=optimize_mean(pre,[w_equal,w_conv,one,*np.eye(len(SSL))])
            w_max,ok_max,msg_max,inner_max_scores,tmax=optimize_maximin(pre,base_scores,[w_equal,w_conv,w_mean,one,*np.eye(len(SSL))])

            enroll,eval_idx=split_enrollment(q,outer_idx,K,f"outer|{outer}|{target}")
            Pouter=np.column_stack([pred_cache[((outer,),n)][eval_idx,j] for n in SSL])
            pred_map={
                "nested_best_single":pred_cache[((outer,),best)][eval_idx,j],
                "equal_ssl_ensemble":Pouter@w_equal,
                "inner_convex_ssl_ensemble":Pouter@w_conv,
                "component_balanced_mean":Pouter@w_mean,
                "component_balanced_maximin":Pouter@w_max,
            }
            for m,pred in pred_map.items():
                for ii,pv in zip(eval_idx,pred):
                    out.append({"sample_id":q.loc[ii,"sample_id"],"speaker_id":q.loc[ii,"speaker_id"],
                                "session_id":outer,"target":target,"method":m,
                                "y_true":float(Y[ii,j]),"pred":float(pv)})
            sel.append({
                "outer_session":outer,"target":target,"best_single":best,
                "best_inner_overall":float(base_scores[0]),"best_inner_between":float(base_scores[1]),"best_inner_within":float(base_scores[2]),
                "convex_ok":bool(ok_conv),"convex_message":msg_conv,
                "convex_weights":json.dumps({n:float(w) for n,w in zip(SSL,w_conv)},sort_keys=True),
                "mean_ok":bool(ok_mean),"mean_message":msg_mean,
                "mean_weights":json.dumps({n:float(w) for n,w in zip(SSL,w_mean)},sort_keys=True),
                "mean_inner_overall":float(inner_mean_scores[0]),"mean_inner_between":float(inner_mean_scores[1]),"mean_inner_within":float(inner_mean_scores[2]),
                "maximin_ok":bool(ok_max),"maximin_message":msg_max,
                "maximin_weights":json.dumps({n:float(w) for n,w in zip(SSL,w_max)},sort_keys=True),
                "maximin_inner_t":float(tmax),"maximin_inner_overall":float(inner_max_scores[0]),
                "maximin_inner_between":float(inner_max_scores[1]),"maximin_inner_within":float(inner_max_scores[2]),
            })
            print("SELECT",outer,target,"best",best,"t",tmax,"max",w_max,flush=True)

    O=pd.DataFrame(out)
    O.to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(fit_rows).to_csv(root/"training_audit.csv",index=False)
    pd.DataFrame(sel).to_csv(root/"selection_audit.csv",index=False)

    metric_rows=[]; delta_rows=[]; ctrl_rows=[]
    rng=np.random.default_rng(int(cfg["bootstrap_seed"])); reps=int(cfg["bootstrap_reps"])
    for target in ["arousal","dominance"]:
        zt=O[O.target==target]; aligned={}; boot={}; speakers=None
        for m in methods:
            sp,mm=speaker_moments(zt[zt.method==m])
            if speakers is None: speakers=sp
            elif sp!=speakers: raise RuntimeError("speaker mismatch")
            aligned[m]=mm
        ns=len(speakers); counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
        for m,mm in aligned.items():
            boot[m]={}
            for met in ["overall","between","within"]:
                vals=ccc_from_mom((counts@mm[met])/ns); boot[m][met]=vals
                lo,hi=np.nanquantile(vals,[.025,.975])
                metric_rows.append({"target":target,"method":m,"metric":met,
                                    "ccc":float(ccc_from_mom(mm[met].mean(0))),
                                    "ci95_low":float(lo),"ci95_high":float(hi)})
        base="nested_best_single"
        for m in methods:
            if m==base: continue
            for met in ["overall","between","within"]:
                dv=boot[m][met]-boot[base][met]; lo,hi=np.nanquantile(dv,[.025,.975])
                pt=float(ccc_from_mom(aligned[m][met].mean(0))-ccc_from_mom(aligned[base][met].mean(0)))
                delta_rows.append({"target":target,"method":m,"reference":base,"metric":met,
                                   "delta_ccc":pt,"ci95_low":float(lo),"ci95_high":float(hi),
                                   "ci_excludes_zero":bool(lo>0 or hi<0)})
        primary="component_balanced_maximin"
        for ref in ["inner_convex_ssl_ensemble","component_balanced_mean"]:
            for met in ["overall","between","within"]:
                dv=boot[primary][met]-boot[ref][met]; lo,hi=np.nanquantile(dv,[.025,.975])
                pt=float(ccc_from_mom(aligned[primary][met].mean(0))-ccc_from_mom(aligned[ref][met].mean(0)))
                ctrl_rows.append({"target":target,"method":primary,"reference":ref,"metric":met,
                                  "delta_ccc":pt,"ci95_low":float(lo),"ci95_high":float(hi),
                                  "ci_excludes_zero":bool(lo>0 or hi<0)})
    M=pd.DataFrame(metric_rows); D=pd.DataFrame(delta_rows); C=pd.DataFrame(ctrl_rows)
    M.to_csv(root/"component_metrics.csv",index=False)
    D.to_csv(root/"deltas_vs_best_single.csv",index=False)
    C.to_csv(root/"maximin_vs_controls.csv",index=False)

    prim=D[D.method=="component_balanced_maximin"]
    positive=[]
    for t,g in prim.groupby("target"):
        dd={r.metric:r.delta_ccc for r in g.itertuples()}
        if all(dd.get(k,-1)>0 for k in ["overall","between","within"]): positive.append(t)
    audit={
        "experiment_id":cfg["experiment_id"],"validity":"valid","decision":"pass" if positive else "mixed",
        "task":"raw absolute IEMOCAP Arousal/Dominance","outer_split":"5-session LOSO",
        "inner_split":"nested session LOSO","common_samples":int(len(q)),"speakers":int(q.speaker_id.nunique()),
        "evaluation_enrollment_k":K,"positive_all_three_targets_for_maximin":positive,
        "fit_warning_count_total":int(pd.DataFrame(fit_rows).warning_count.sum()),
        "selection_leakage_check":"pass by construction: pair-held fits exclude both outer and inner validation sessions",
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "ridge_alpha":cfg["ridge_alpha"],"ridge_solver":cfg["ridge_solver"],"bootstrap_reps":reps,
        "bootstrap_unit":"speaker","objective":"maximin Between/Within inner gain with nondecreasing Overall constraint"
    },indent=2)+"\n")
    print("\nMAXIMIN DELTAS")
    print(prim.to_string(index=False))
    print("\nMAXIMIN VS CONTROLS")
    print(C.to_string(index=False))
    print("\nAUDIT",json.dumps(audit,indent=2))

if __name__=="__main__": main()
