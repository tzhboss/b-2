#!/usr/bin/env python3
from __future__ import annotations
import argparse,glob,json,re,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_iemocap_paired_crossmodel_component_attribution_20260924 import ccc,moments,point,qci

def speaker_id(x):
    m=re.match(r"(Ses\d\d)[FM]_.+_([FM])\d+\.wav$",str(x))
    if not m: raise ValueError(x)
    return m.group(1)+"_"+m.group(2)

def session_id(x):
    m=re.match(r"(Ses\d\d)[FM]_",str(x))
    if not m: raise ValueError(x)
    return m.group(1)

def weights(df):
    c=df.speaker_id.astype(str).value_counts(); w=df.speaker_id.astype(str).map(lambda s:1/c[s]).to_numpy(float)
    return w/w.mean()

def fit(xtr,xte,ytr,w,alpha,solver="auto"):
    sc=StandardScaler(); sc.fit(xtr,sample_weight=w); m=Ridge(alpha=alpha,solver=solver)
    m.fit(sc.transform(xtr),ytr,sample_weight=w); return m.predict(sc.transform(xte))

def load_simple(pattern):
    ids=[]; arr=[]
    for f in sorted(glob.glob(pattern)):
        z=np.load(f); ids.extend(z["sample_id"].astype(str)); arr.append(z["embedding"].astype(np.float32))
    return pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))}),np.concatenate(arr)

def load_wavlm(pattern):
    ids=[]; arr=[]; layers=None
    for f in sorted(glob.glob(pattern)):
        z=np.load(f); ids.extend(z["sample_id"].astype(str)); arr.append(z["embedding"].astype(np.float32)); layers=z["layers"].tolist()
    a=np.concatenate(arr)
    return {f"wavlm_layer{int(l)}":(pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))}),a[:,i,:]) for i,l in enumerate(layers)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)
    labs=[pd.read_parquet(f,columns=["file","EmoAct","EmoDom"]) for f in sorted(glob.glob(cfg["dataset_glob"]))]
    d=pd.concat(labs,ignore_index=True).dropna().copy(); d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(speaker_id); d["session_id"]=d.file.map(session_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})
    speakers=sorted(d.speaker_id.unique()); sessions=sorted(d.session_id.unique())
    if sessions!=["Ses01","Ses02","Ses03","Ses04","Ses05"]: raise RuntimeError(f"unexpected sessions {sessions}")
    fmap={s:i for i,s in enumerate(sessions)}
    solver=str(cfg.get("ridge_solver","auto"))
    models=load_wavlm(cfg["embeddings"]["wavlm"])
    for name,pat in cfg["embeddings"].items():
        if name=="wavlm": continue
        models[name]=load_simple(pat)
    oof_rows=[]; metric_rows=[]; delta_rows=[]; contrast_rows=[]; rank_rows=[]; center_rows=[]; rng=np.random.default_rng(int(cfg["bootstrap_seed"]))
    comps=[("relative_minus_absolute","relative","absolute"),("hybrid_minus_absolute","hybrid","absolute"),("hybrid_minus_relative","hybrid","relative")]
    for model_name,(em,H0) in models.items():
        if em.sample_id.duplicated().any(): raise RuntimeError(f"duplicate ids {model_name}")
        q=d.merge(em,on="sample_id",validate="one_to_one")
        if len(q)!=10039: raise RuntimeError(f"coverage {model_name}={len(q)}")
        H=H0[q.row.to_numpy()]; sf=q.session_id.map(fmap).to_numpy(); Y=q[["arousal","dominance"]].to_numpy(float)
        centers={sp:np.median(H[q.speaker_id.to_numpy()==sp],axis=0) for sp in speakers}
        C=np.vstack([centers[sp] for sp in q.speaker_id]); R=H-C
        feats={"absolute":H,"relative":R,"hybrid":np.concatenate([R,C],axis=1)}
        preds={m:np.empty_like(Y) for m in feats}
        for fold in range(int(cfg["outer_folds"])):
            tr=sf!=fold; te=sf==fold; w=weights(q.loc[tr])
            for meth,X in feats.items(): preds[meth][te]=fit(X[tr],X[te],Y[tr],w,float(cfg["ridge_alpha"]),solver)
        center_rows.append({"model":model_name,"rows":len(q),"speakers":len(speakers),"embedding_dim":H.shape[1],"hybrid_dim":feats["hybrid"].shape[1],"median_center_norm":float(np.median([np.linalg.norm(v) for v in centers.values()]))})
        for j,target in enumerate(["arousal","dominance"]):
            aligned={}; boot={}; ns=len(speakers); counts=rng.multinomial(ns,np.full(ns,1/ns),size=int(cfg["bootstrap_reps"])).astype(float)
            for meth in ["absolute","relative","hybrid"]:
                z=pd.DataFrame({"sample_id":q.sample_id,"speaker_id":q.speaker_id,"target":target,"y_true":Y[:,j],"pred":preds[meth][:,j],"fold":sf})
                sp,mm=moments(z); aligned[meth]=(sp,mm); boot[meth]={}
                for metric in ["overall","between","within"]:
                    vals=ccc((counts@mm[metric])/ns); boot[meth][metric]=vals; lo,hi=qci(vals)
                    metric_rows.append({"model":model_name,"target":target,"method":meth,"metric":metric,"ccc":point(mm[metric]),"ci95_low":lo,"ci95_high":hi})
                for row in z.itertuples(index=False):
                    oof_rows.append({"sample_id":row.sample_id,"speaker_id":row.speaker_id,"model":model_name,"method":meth,"target":target,"y_true":row.y_true,"pred":row.pred,"fold":row.fold})
            for metric in ["overall","between","within"]:
                scores={m:point(aligned[m][1][metric]) for m in aligned}
                rank_rows.append({"model":model_name,"target":target,"metric":metric,"ranking":" > ".join(sorted(scores,key=scores.get,reverse=True)),**scores})
                for cname,x,y in comps:
                    dv=boot[x][metric]-boot[y][metric]; lo,hi=qci(dv)
                    delta_rows.append({"model":model_name,"target":target,"comparison":cname,"metric":metric,"delta_ccc":scores[x]-scores[y],"ci95_low":lo,"ci95_high":hi,"ci_excludes_zero":bool(lo>0 or hi<0)})
            for cname,x,y in comps:
                for contrast,m1,m2 in [("between_minus_within","between","within"),("overall_minus_within","overall","within")]:
                    dv=(boot[x][m1]-boot[y][m1])-(boot[x][m2]-boot[y][m2]); lo,hi=qci(dv)
                    pt=(point(aligned[x][1][m1])-point(aligned[y][1][m1]))-(point(aligned[x][1][m2])-point(aligned[y][1][m2]))
                    contrast_rows.append({"model":model_name,"target":target,"comparison":cname,"contrast":contrast,"delta_of_deltas":pt,"ci95_low":lo,"ci95_high":hi,"ci_excludes_zero":bool(lo>0 or hi<0)})
    pd.DataFrame(oof_rows).to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(metric_rows).to_csv(root/"component_metrics.csv",index=False)
    pd.DataFrame(delta_rows).to_csv(root/"method_component_deltas.csv",index=False)
    pd.DataFrame(contrast_rows).to_csv(root/"method_profile_contrasts.csv",index=False)
    pd.DataFrame(rank_rows).to_csv(root/"rank_summary.csv",index=False)
    pd.DataFrame(center_rows).to_csv(root/"center_audit.csv",index=False)
    meta={"experiment_id":cfg["experiment_id"],"task":"raw absolute VAD","relative_role":"diagnostic representation only","center":"full-speaker unlabeled coordinate-wise median (oracle diagnostic)","fold_policy":"strict leave-one-session-out","sessions":sessions,"ridge_solver":solver,"bootstrap_unit":"speaker","bootstrap_reps":cfg["bootstrap_reps"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    (root/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    c=pd.DataFrame(contrast_rows); print(c[c.ci_excludes_zero].to_string(index=False))
if __name__=="__main__": main()

