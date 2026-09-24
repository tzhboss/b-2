#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, hashlib, itertools, json, math, re, subprocess, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def speaker_id(x):
    m=re.match(r"(Ses\d\d)[FM]_.+_([FM])\d+\.wav$",str(x))
    if not m: raise ValueError(x)
    return m.group(1)+"_"+m.group(2)

def session_id(x):
    m=re.match(r"(Ses\d\d)[FM]_",str(x))
    if not m: raise ValueError(x)
    return m.group(1)

def ccc_from_mom(m):
    m=np.asarray(m,float)
    y,p,y2,p2,yp=[m[...,i] for i in range(5)]
    vy=np.maximum(y2-y*y,0); vp=np.maximum(p2-p*p,0)
    den=vy+vp+(y-p)**2
    return np.divide(2*(yp-y*p),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)

def speaker_moments(df):
    rows=[]
    for sp,g in df.groupby("speaker_id",sort=True):
        y=g.y_true.to_numpy(float); p=g.pred.to_numpy(float)
        rows.append((str(sp),np.array([y.mean(),p.mean(),np.mean(y*y),np.mean(p*p),np.mean(y*p)],float)))
    s=[x[0] for x in rows]; M=np.vstack([x[1] for x in rows])
    between=np.c_[M[:,0],M[:,1],M[:,0]**2,M[:,1]**2,M[:,0]*M[:,1]]
    within=np.c_[np.zeros(len(M)),np.zeros(len(M)),M[:,2]-M[:,0]**2,M[:,3]-M[:,1]**2,M[:,4]-M[:,0]*M[:,1]]
    return s,{"overall":M,"between":between,"within":within}

def component_scores(df):
    _,mm=speaker_moments(df)
    return {k:float(ccc_from_mom(v.mean(0))) for k,v in mm.items()}

def equal_speaker_weights(speakers):
    s=pd.Series(np.asarray(speakers).astype(str))
    c=s.value_counts()
    w=s.map(lambda x:1.0/c[x]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else float("nan")

def load_simple(pattern):
    ids=[]; blocks=[]
    for f in sorted(glob.glob(pattern)):
        z=np.load(f)
        ids.extend(z["sample_id"].astype(str))
        blocks.append(z["embedding"].astype(np.float32))
    if not blocks: raise RuntimeError(f"no embeddings: {pattern}")
    arr=np.concatenate(blocks)
    tab=pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))})
    if tab.sample_id.duplicated().any(): raise RuntimeError(f"duplicate embedding ids: {pattern}")
    return tab,arr

def load_wavlm(pattern):
    ids=[]; blocks=[]; layers=None
    for f in sorted(glob.glob(pattern)):
        z=np.load(f)
        ids.extend(z["sample_id"].astype(str))
        blocks.append(z["embedding"].astype(np.float32))
        layers=z["layers"].tolist()
    if not blocks: raise RuntimeError("no WavLM embeddings")
    arr=np.concatenate(blocks)
    tab=pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))})
    if tab.sample_id.duplicated().any(): raise RuntimeError("duplicate WavLM ids")
    return {f"wavlm_layer{int(l)}":(tab.copy(),arr[:,i,:]) for i,l in enumerate(layers)}

def align_matrix(q,tab,arr):
    ix=tab.set_index("sample_id").loc[q.sample_id.astype(str),"row"].to_numpy(int)
    return arr[ix].astype(np.float32,copy=False)

def fit_fold(X,Y,q,tr,te,alpha,solver,quadratic):
    w=equal_speaker_weights(q.loc[tr,"speaker_id"])
    sc=StandardScaler()
    sc.fit(X[tr],sample_weight=w)
    A=sc.transform(X[tr]); B=sc.transform(X[te])
    if quadratic:
        poly=PolynomialFeatures(degree=2,include_bias=False)
        A=poly.fit_transform(A); B=poly.transform(B)
        sc2=StandardScaler()
        sc2.fit(A,sample_weight=w)
        A=sc2.transform(A); B=sc2.transform(B)
    mdl=Ridge(alpha=alpha,solver=solver)
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always")
        mdl.fit(A,Y[tr],sample_weight=w)
    pred=mdl.predict(B)
    # Retrieval context uses the common pre-polynomial standardized representation.
    Ztr=sc.transform(X[tr])
    speakers=sorted(q.loc[tr,"speaker_id"].astype(str).unique())
    centroids=np.vstack([Ztr[q.loc[tr,"speaker_id"].astype(str).to_numpy()==sp].mean(0) for sp in speakers])
    stats={}
    for j,t in enumerate(["arousal","dominance"]):
        stats[t]=np.array([[Y[tr][q.loc[tr,"speaker_id"].astype(str).to_numpy()==sp,j].mean(),
                            Y[tr][q.loc[tr,"speaker_id"].astype(str).to_numpy()==sp,j].std(ddof=0)]
                           for sp in speakers],float)
    ctx={"scaler":sc,"train_speakers":speakers,"centroids":centroids,"stats":stats}
    return pred,ctx,len(ww)

def stable_order(ids,salt):
    def h(x):
        return hashlib.sha256((str(salt)+"|"+str(x)).encode()).digest()
    return sorted(range(len(ids)),key=lambda i:h(ids[i]))

def split_enrollment(q,idx,k,salt):
    idx=np.asarray(idx,int)
    enroll=[]; evaluate=[]
    for sp in sorted(q.loc[idx,"speaker_id"].astype(str).unique()):
        ii=idx[q.loc[idx,"speaker_id"].astype(str).to_numpy()==sp]
        ids=q.loc[ii,"sample_id"].astype(str).tolist()
        order=stable_order(ids,f"{salt}|{sp}")
        kk=min(int(k),max(1,len(ii)-1))
        es=set(ii[np.asarray(order[:kk],int)].tolist())
        enroll.extend(sorted(es))
        evaluate.extend([int(x) for x in ii if int(x) not in es])
    return np.asarray(sorted(enroll),int),np.asarray(sorted(evaluate),int)

def retrieve_stats(q,X,ctx,enroll_idx,target,k):
    sc=ctx["scaler"]
    v=sc.transform(X[enroll_idx]).mean(0)
    C=ctx["centroids"]
    den=np.maximum(np.linalg.norm(C,axis=1),1e-12)*max(float(np.linalg.norm(v)),1e-12)
    sim=(C@v)/den
    order=np.argsort(-sim)[:min(int(k),len(sim))]
    st=ctx["stats"][target][order]
    return float(st[:,0].mean()),float(st[:,1].mean()),[ctx["train_speakers"][i] for i in order]

def calibrate_speaker(q,X,ctx,pred_full,idx,target,j,k_enroll,k_neighbors,salt):
    enroll,evaluate=split_enrollment(q,idx,k_enroll,salt)
    out={}
    for sp in sorted(q.loc[idx,"speaker_id"].astype(str).unique()):
        ee=enroll[q.loc[enroll,"speaker_id"].astype(str).to_numpy()==sp]
        vv=evaluate[q.loc[evaluate,"speaker_id"].astype(str).to_numpy()==sp]
        mu,sig,neighbors=retrieve_stats(q,X,ctx,ee,target,k_neighbors)
        pe=pred_full[ee,j]; pv=pred_full[vv,j]
        pm=float(pe.mean()); ps=max(float(pe.std(ddof=0)),1e-8)
        out[sp]={
            "eval_idx":vv,
            "enroll_idx":ee,
            "mu":mu,"sig":sig,"pm":pm,"ps":ps,"neighbors":neighbors,
            "mu_pred":pv-pm+mu,
            "sigma_pred":(pv-pm)/ps*sig+pm,
            "mu_sigma_pred":(pv-pm)/ps*sig+mu,
        }
    return out,enroll,evaluate

def make_df(q,Y,target_idx,indices,pred):
    return pd.DataFrame({
        "sample_id":q.loc[indices,"sample_id"].astype(str).to_numpy(),
        "speaker_id":q.loc[indices,"speaker_id"].astype(str).to_numpy(),
        "session_id":q.loc[indices,"session_id"].astype(str).to_numpy(),
        "y_true":Y[indices,target_idx],
        "pred":np.asarray(pred,float),
    })

def simplex_weights(y,P,speakers):
    m=P.shape[1]
    wobs=equal_speaker_weights(speakers)
    fun=lambda w:-weighted_ccc(y,P@w,wobs)
    cons={"type":"eq","fun":lambda w:np.sum(w)-1.0}
    res=minimize(fun,np.full(m,1/m),method="SLSQP",bounds=[(0.0,1.0)]*m,constraints=[cons],
                 options={"maxiter":500,"ftol":1e-10})
    if not res.success or not np.all(np.isfinite(res.x)):
        return np.full(m,1/m),False,str(res.message)
    w=np.maximum(res.x,0); w=w/max(w.sum(),1e-12)
    return w,True,str(res.message)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)

    cols=["file","EmoAct","EmoDom","speaking_rate","pitch_mean"]
    d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(cfg["dataset_glob"]))],
                ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str)
    d["speaker_id"]=d.file.map(speaker_id)
    d["session_id"]=d.file.map(session_id)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float))
    d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})

    emb=load_wavlm(cfg["embeddings"]["wavlm"])
    for name in ["emotion2vec_plus_large","hubert_base_ls960","wav2vec2_base"]:
        emb[name]=load_simple(cfg["embeddings"][name])
    common=set(d.sample_id.astype(str))
    for tab,_ in emb.values(): common &= set(tab.sample_id.astype(str))
    q=d[d.sample_id.astype(str).isin(common)].sort_values("sample_id").reset_index(drop=True)
    if len(q)!=10039: raise RuntimeError(f"unexpected common coverage {len(q)}")
    sessions=sorted(q.session_id.unique())
    if sessions!=["Ses01","Ses02","Ses03","Ses04","Ses05"]: raise RuntimeError(sessions)
    Y=q[["arousal","dominance"]].to_numpy(float)

    pros=q[["pitch_abs","rate_abs"]].to_numpy(np.float32)
    models={
        "pitch_rate_linear":(pros,False),
        "pitch_rate_quadratic":(pros,True),
    }
    for name,(tab,arr) in emb.items(): models[name]=(align_matrix(q,tab,arr),False)
    ssl=["wavlm_layer12","wavlm_layer24","emotion2vec_plus_large","hubert_base_ls960","wav2vec2_base"]
    model_names=list(models)

    # Cache fits for every outer held session and every unordered pair used by nested inner LOSO.
    pred_cache={}; ctx_cache={}; fit_rows=[]
    held_sets=[(s,) for s in sessions]+list(itertools.combinations(sessions,2))
    alpha=float(cfg["ridge_alpha"]); solver=str(cfg["ridge_solver"])
    for held in held_sets:
        tr=~q.session_id.isin(held).to_numpy()
        te=~tr
        if set(q.loc[tr,"speaker_id"]) & set(q.loc[te,"speaker_id"]): raise RuntimeError(f"speaker leakage {held}")
        for name,(X,quad) in models.items():
            pred,ctx,nwarn=fit_fold(X,Y,q,tr,te,alpha,solver,quad)
            full=np.full((len(q),2),np.nan,float); full[te]=pred
            key=(tuple(held),name)
            pred_cache[key]=full; ctx_cache[key]=ctx
            fit_rows.append({"held_sessions":"+".join(held),"model":name,"train_rows":int(tr.sum()),
                             "test_rows":int(te.sum()),"train_speakers":int(q.loc[tr,"speaker_id"].nunique()),
                             "test_speakers":int(q.loc[te,"speaker_id"].nunique()),"warning_count":int(nwarn)})
            print("FIT",held,name,"warnings",nwarn,flush=True)

    methods=["nested_best_single","equal_ssl_ensemble","inner_convex_ssl_ensemble",
             "pldc_mu","pldc_sigma","pldc_mu_sigma",
             "component_aware","component_aware_pldc_mu","component_aware_pldc_mu_sigma"]
    out_rows=[]; select_rows=[]; retrieval_rows=[]
    K=int(cfg["enrollment_k"]); k_candidates=[int(x) for x in cfg["pldc_neighbor_k_candidates"]]

    for outer in sessions:
        outer_train=[s for s in sessions if s!=outer]
        outer_idx=np.where(q.session_id.to_numpy()==outer)[0]
        # Inner OOF predictions use pair-held fits: one is outer test, the other is inner validation.
        inner_pred={name:np.full((len(q),2),np.nan,float) for name in model_names}
        for inner in outer_train:
            keyheld=tuple(sorted((outer,inner)))
            vi=np.where(q.session_id.to_numpy()==inner)[0]
            for name in model_names:
                inner_pred[name][vi]=pred_cache[(keyheld,name)][vi]
        inner_idx=np.where(q.session_id.isin(outer_train).to_numpy())[0]
        for j,target in enumerate(["arousal","dominance"]):
            model_scores={}
            for name in model_names:
                z=make_df(q,Y,j,inner_idx,inner_pred[name][inner_idx,j])
                model_scores[name]=component_scores(z)
            best=max(model_names,key=lambda n:model_scores[n]["overall"])
            bex=max(model_names,key=lambda n:model_scores[n]["between"])
            wex=max(model_names,key=lambda n:model_scores[n]["within"])

            Pinner=np.column_stack([inner_pred[n][inner_idx,j] for n in ssl])
            cw,conv_ok,conv_msg=simplex_weights(Y[inner_idx,j],Pinner,q.loc[inner_idx,"speaker_id"])
            equal_inner=Pinner.mean(1)
            convex_inner=Pinner@cw

            # Select PLDC neighbor count using only inner folds and best-single base.
            k_perf={}
            for nk in k_candidates:
                pieces=[]
                for inner in outer_train:
                    keyheld=tuple(sorted((outer,inner)))
                    vi=np.where(q.session_id.to_numpy()==inner)[0]
                    ctx=ctx_cache[(keyheld,best)]
                    X=models[best][0]
                    cal,enr,eva=calibrate_speaker(q,X,ctx,inner_pred[best],vi,target,j,K,nk,
                                                  f"inner|{outer}|{inner}|{target}")
                    pp=[]; ii=[]
                    for sp,v in cal.items():
                        ii.extend(v["eval_idx"].tolist()); pp.extend(v["mu_sigma_pred"].tolist())
                    ii=np.asarray(ii,int); order=np.argsort(ii); ii=ii[order]; pp=np.asarray(pp)[order]
                    pieces.append(make_df(q,Y,j,ii,pp))
                zz=pd.concat(pieces,ignore_index=True)
                k_perf[nk]=component_scores(zz)["overall"]
            nk=max(k_candidates,key=lambda k:k_perf[k])

            # Common label-free enrollment/evaluation subset for outer held session.
            enroll,eval_idx=split_enrollment(q,outer_idx,K,f"outer|{outer}|{target}")
            best_pred=pred_cache[((outer,),best)]
            base_eval=best_pred[eval_idx,j]
            equal_eval=np.column_stack([pred_cache[((outer,),n)][eval_idx,j] for n in ssl]).mean(1)
            convex_eval=np.column_stack([pred_cache[((outer,),n)][eval_idx,j] for n in ssl])@cw

            # Published-style calibration control on nested best-single.
            cal_best,_,_=calibrate_speaker(q,models[best][0],ctx_cache[((outer,),best)],best_pred,
                                           outer_idx,target,j,K,nk,f"outer|{outer}|{target}")
            pldc={"pldc_mu":{},"pldc_sigma":{},"pldc_mu_sigma":{}}
            for sp,v in cal_best.items():
                for meth,keyp in [("pldc_mu","mu_pred"),("pldc_sigma","sigma_pred"),("pldc_mu_sigma","mu_sigma_pred")]:
                    for ii,pv in zip(v["eval_idx"],v[keyp]): pldc[meth][int(ii)]=float(pv)
                retrieval_rows.append({"outer_session":outer,"target":target,"role":"pldc_base",
                                       "speaker_id":sp,"model":best,"neighbor_k":nk,
                                       "neighbors":";".join(v["neighbors"])})

            # Decomposition-guided recomposition.
            pb=pred_cache[((outer,),bex)]; pw=pred_cache[((outer,),wex)]
            ca={m:{} for m in ["component_aware","component_aware_pldc_mu","component_aware_pldc_mu_sigma"]}
            ctxb=ctx_cache[((outer,),bex)]; Xb=models[bex][0]
            for sp in sorted(q.loc[outer_idx,"speaker_id"].astype(str).unique()):
                ee=enroll[q.loc[enroll,"speaker_id"].astype(str).to_numpy()==sp]
                vv=eval_idx[q.loc[eval_idx,"speaker_id"].astype(str).to_numpy()==sp]
                muBpred=float(pb[ee,j].mean()); muWpred=float(pw[ee,j].mean())
                sdW=max(float(pw[ee,j].std(ddof=0)),1e-8)
                muLab,sigLab,neighbors=retrieve_stats(q,Xb,ctxb,ee,target,nk)
                resid=pw[vv,j]-muWpred
                p0=muBpred+resid
                p1=muLab+resid
                p2=muLab+resid/sdW*sigLab
                for ii,a0,a1,a2 in zip(vv,p0,p1,p2):
                    ca["component_aware"][int(ii)]=float(a0)
                    ca["component_aware_pldc_mu"][int(ii)]=float(a1)
                    ca["component_aware_pldc_mu_sigma"][int(ii)]=float(a2)
                retrieval_rows.append({"outer_session":outer,"target":target,"role":"between_expert",
                                       "speaker_id":sp,"model":bex,"neighbor_k":nk,
                                       "neighbors":";".join(neighbors)})

            pred_methods={
                "nested_best_single":dict(zip(eval_idx,base_eval)),
                "equal_ssl_ensemble":dict(zip(eval_idx,equal_eval)),
                "inner_convex_ssl_ensemble":dict(zip(eval_idx,convex_eval)),
                **pldc,**ca,
            }
            for meth,mp in pred_methods.items():
                if set(mp)!=set(map(int,eval_idx)): raise RuntimeError(f"coverage mismatch {outer}/{target}/{meth}")
                for ii in eval_idx:
                    out_rows.append({"sample_id":q.loc[ii,"sample_id"],"speaker_id":q.loc[ii,"speaker_id"],
                                     "session_id":outer,"target":target,"method":meth,
                                     "y_true":float(Y[ii,j]),"pred":float(mp[int(ii)])})

            select_rows.append({
                "outer_session":outer,"target":target,"best_single":best,"between_expert":bex,"within_expert":wex,
                "pldc_neighbor_k":nk,"enrollment_k":K,"convex_ok":bool(conv_ok),"convex_message":conv_msg,
                "convex_weights":json.dumps({n:float(w) for n,w in zip(ssl,cw)},sort_keys=True),
                "inner_best_overall":float(model_scores[best]["overall"]),
                "inner_between_expert_ccc":float(model_scores[bex]["between"]),
                "inner_within_expert_ccc":float(model_scores[wex]["within"]),
                "inner_equal_overall":float(weighted_ccc(Y[inner_idx,j],equal_inner,
                    equal_speaker_weights(q.loc[inner_idx,"speaker_id"]))),
                "inner_convex_overall":float(weighted_ccc(Y[inner_idx,j],convex_inner,
                    equal_speaker_weights(q.loc[inner_idx,"speaker_id"]))),
                "pldc_k_scores":json.dumps({str(k):float(v) for k,v in k_perf.items()},sort_keys=True),
            })
            print("SELECT",outer,target,"best",best,"B",bex,"W",wex,"k",nk,flush=True)

    O=pd.DataFrame(out_rows)
    O.to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(fit_rows).to_csv(root/"training_audit.csv",index=False)
    pd.DataFrame(select_rows).to_csv(root/"selection_audit.csv",index=False)
    pd.DataFrame(retrieval_rows).to_csv(root/"retrieval_audit.csv",index=False)

    metric_rows=[]; delta_rows=[]; control_rows=[]
    rng=np.random.default_rng(int(cfg["bootstrap_seed"]))
    reps=int(cfg["bootstrap_reps"])
    for target in ["arousal","dominance"]:
        zt=O[O.target==target]
        aligned={}; boot={}
        speakers=None
        for meth in methods:
            zz=zt[zt.method==meth].copy()
            sp,mm=speaker_moments(zz)
            if speakers is None: speakers=sp
            elif sp!=speakers: raise RuntimeError("speaker alignment mismatch")
            aligned[meth]=mm
        ns=len(speakers)
        counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
        for meth,mm in aligned.items():
            boot[meth]={}
            for metric in ["overall","between","within"]:
                vals=ccc_from_mom((counts@mm[metric])/ns)
                boot[meth][metric]=vals
                lo,hi=np.nanquantile(vals,[.025,.975])
                metric_rows.append({"target":target,"method":meth,"metric":metric,
                                    "ccc":float(ccc_from_mom(mm[metric].mean(0))),
                                    "ci95_low":float(lo),"ci95_high":float(hi)})
        base="nested_best_single"
        for meth in methods:
            if meth==base: continue
            for metric in ["overall","between","within"]:
                dv=boot[meth][metric]-boot[base][metric]
                lo,hi=np.nanquantile(dv,[.025,.975])
                pt=float(ccc_from_mom(aligned[meth][metric].mean(0))-ccc_from_mom(aligned[base][metric].mean(0)))
                delta_rows.append({"target":target,"method":meth,"reference":base,"metric":metric,
                                   "delta_ccc":pt,"ci95_low":float(lo),"ci95_high":float(hi),
                                   "ci_excludes_zero":bool(lo>0 or hi<0)})
        primary="component_aware_pldc_mu_sigma"
        for ref in ["equal_ssl_ensemble","inner_convex_ssl_ensemble","pldc_mu_sigma"]:
            for metric in ["overall","between","within"]:
                dv=boot[primary][metric]-boot[ref][metric]
                lo,hi=np.nanquantile(dv,[.025,.975])
                pt=float(ccc_from_mom(aligned[primary][metric].mean(0))-ccc_from_mom(aligned[ref][metric].mean(0)))
                control_rows.append({"target":target,"method":primary,"reference":ref,"metric":metric,
                                     "delta_ccc":pt,"ci95_low":float(lo),"ci95_high":float(hi),
                                     "ci_excludes_zero":bool(lo>0 or hi<0)})

    M=pd.DataFrame(metric_rows); D=pd.DataFrame(delta_rows); C=pd.DataFrame(control_rows)
    M.to_csv(root/"component_metrics.csv",index=False)
    D.to_csv(root/"deltas_vs_best_single.csv",index=False)
    C.to_csv(root/"primary_vs_strong_controls.csv",index=False)

    p=D[D.method=="component_aware_pldc_mu_sigma"]
    positive_targets=[]
    for t,g in p.groupby("target"):
        dd={r.metric:r.delta_ccc for r in g.itertuples()}
        if all(dd.get(m,-1)>0 for m in ["overall","between","within"]): positive_targets.append(t)
    decision="pass" if positive_targets else "mixed"
    audit={
        "experiment_id":cfg["experiment_id"],"validity":"valid","decision":decision,
        "task":"raw absolute IEMOCAP Arousal/Dominance","outer_split":"5-session LOSO",
        "inner_split":"nested session LOSO within each outer training set",
        "common_samples":int(len(q)),"speakers":int(q.speaker_id.nunique()),
        "primary_enrollment_k":K,"evaluation_rows_per_target":int(len(O[O.target=="arousal"])/len(methods)),
        "methods":methods,"positive_all_three_targets_for_primary":positive_targets,
        "no_outer_selection_leakage":"by construction; all selection uses pair-held inner predictions excluding the outer session",
        "fit_warning_count_total":int(pd.DataFrame(fit_rows).warning_count.sum()),
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "ridge_alpha":alpha,"ridge_solver":solver,"bootstrap_reps":reps,"bootstrap_unit":"speaker",
        "relative_role":"diagnostic/recomposition only; absolute labels remain the task",
        "pldc_note":"common-representation published-style reimplementation, not claimed as exact-paper reproduction",
    },indent=2)+"\n")
    print("\nPRIMARY DELTAS VS BEST SINGLE")
    print(p.to_string(index=False))
    print("\nPRIMARY VS STRONG CONTROLS")
    print(C.to_string(index=False))
    print("\nAUDIT",json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
