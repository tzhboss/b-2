#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, itertools, json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_iemocap_paired_crossmodel_component_attribution_20260924 import ccc, moments, point, qci

def speaker_id(x):
    m=re.match(r"(Ses\d\d)[FM]_.+_([FM])\d+\.wav$",str(x))
    if not m: raise ValueError(x)
    return m.group(1)+"_"+m.group(2)

def session_id(x):
    m=re.match(r"(Ses\d\d)[FM]_",str(x))
    if not m: raise ValueError(x)
    return m.group(1)

def weights(df):
    c=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def fit_predict(xtr,xte,ytr,wtr,alpha,quadratic=False,solver="auto"):
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    a,b=sc.transform(xtr),sc.transform(xte)
    if quadratic:
        poly=PolynomialFeatures(degree=2,include_bias=False)
        a,b=poly.fit_transform(a),poly.transform(b)
        sc2=StandardScaler(); sc2.fit(a,sample_weight=wtr)
        a,b=sc2.transform(a),sc2.transform(b)
    m=Ridge(alpha=alpha,solver=solver); m.fit(a,ytr,sample_weight=wtr)
    return m.predict(b)

def load_simple(pattern):
    ids,blocks=[],[]
    for f in sorted(glob.glob(pattern)):
        z=np.load(f); ids.extend(z["sample_id"].astype(str)); blocks.append(z["embedding"].astype(np.float32))
    if not blocks: raise RuntimeError(f"no embeddings for {pattern}")
    arr=np.concatenate(blocks)
    tab=pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))})
    if tab.sample_id.duplicated().any(): raise RuntimeError(f"duplicate ids for {pattern}")
    return tab,arr

def load_wavlm(pattern):
    ids,blocks,layers=[],[],None
    for f in sorted(glob.glob(pattern)):
        z=np.load(f); ids.extend(z["sample_id"].astype(str)); blocks.append(z["embedding"].astype(np.float32)); layers=z["layers"].tolist()
    if not blocks: raise RuntimeError("no WavLM embeddings")
    arr=np.concatenate(blocks)
    tab=pd.DataFrame({"sample_id":ids,"row":np.arange(len(ids))})
    if tab.sample_id.duplicated().any(): raise RuntimeError("duplicate WavLM ids")
    return {f"wavlm_layer{int(l)}":(tab.copy(),arr[:,i,:]) for i,l in enumerate(layers)}

def align_matrix(q,tab,arr,name):
    idx=tab.set_index("sample_id").loc[q.sample_id.astype(str),"row"].to_numpy(int)
    x=arr[idx]
    if len(x)!=len(q): raise RuntimeError(f"alignment length {name}")
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True,exist_ok=True)
    cols=["file","EmoAct","EmoDom","speaking_rate","pitch_mean"]
    frames=[pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(cfg["dataset_glob"]))]
    d=pd.concat(frames,ignore_index=True).dropna().copy()
    d=d[(d.pitch_mean>0)&(d.speaking_rate>0)].copy()
    d["sample_id"]=d.file.astype(str); d["speaker_id"]=d.file.map(speaker_id); d["session_id"]=d.file.map(session_id)
    d["pitch_abs"]=12*np.log2(d.pitch_mean.astype(float)); d["rate_abs"]=np.log(d.speaking_rate.astype(float))
    d=d.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})
    sessions=sorted(d.session_id.unique())
    if sessions!=["Ses01","Ses02","Ses03","Ses04","Ses05"]: raise RuntimeError(f"unexpected sessions {sessions}")
    emb=load_wavlm(cfg["embeddings"]["wavlm"])
    for name in ["emotion2vec_plus_large","hubert_base_ls960","wav2vec2_base"]:
        emb[name]=load_simple(cfg["embeddings"][name])
    common=set(d.sample_id.astype(str))
    for tab,_ in emb.values(): common &= set(tab.sample_id.astype(str))
    q=d[d.sample_id.astype(str).isin(common)].sort_values("sample_id").reset_index(drop=True)
    if q.sample_id.duplicated().any(): raise RuntimeError("duplicate common sample ids")
    if q.speaker_id.nunique()!=10 or q.session_id.nunique()!=5: raise RuntimeError("unexpected speaker/session coverage")
    prosody=q[["pitch_abs","rate_abs"]].to_numpy(np.float32)
    models={"pitch_rate_linear":(prosody,False),"pitch_rate_quadratic":(prosody,True)}
    for name,(tab,arr) in emb.items(): models[name]=(align_matrix(q,tab,arr,name),False)
    Y=q[["arousal","dominance"]].to_numpy(float)
    preds={name:np.empty_like(Y) for name in models}; audit=[]; alpha=float(cfg["ridge_alpha"]); solver=str(cfg.get("ridge_solver","auto"))
    for fold,held in enumerate(sessions):
        tr=q.session_id.to_numpy()!=held; te=~tr
        trs=set(q.loc[tr,"session_id"]); tes=set(q.loc[te,"session_id"])
        trp=set(q.loc[tr,"speaker_id"]); tep=set(q.loc[te,"speaker_id"])
        if trs & tes or trp & tep: raise RuntimeError(f"split leakage {held}")
        w=weights(q.loc[tr])
        for name,(X,quadratic) in models.items():
            preds[name][te]=fit_predict(X[tr],X[te],Y[tr],w,alpha,quadratic,solver)
            for target in ["arousal","dominance"]:
                audit.append({"model":name,"target":target,"fold":fold,"heldout_session":held,
                              "train_rows":int(tr.sum()),"test_rows":int(te.sum()),
                              "train_sessions":len(trs),"test_sessions":len(tes),
                              "train_speakers":len(trp),"test_speakers":len(tep),
                              "session_overlap":0,"speaker_overlap":0})
    oof=[]; metrics=[]; deltas=[]; contrasts=[]; ranks=[]; speakers_ref=None
    rng=np.random.default_rng(int(cfg["bootstrap_seed"])); reps=int(cfg["bootstrap_reps"])
    for j,target in enumerate(["arousal","dominance"]):
        aligned={}; boot={}
        for name in models:
            z=pd.DataFrame({"sample_id":q.sample_id,"speaker_id":q.speaker_id,"session_id":q.session_id,
                            "target":target,"y_true":Y[:,j],"pred":preds[name][:,j]})
            sp,mm=moments(z)
            if speakers_ref is None: speakers_ref=sp
            elif sp!=speakers_ref: raise RuntimeError("speaker ordering mismatch")
            aligned[name]=(sp,mm)
            for row in z.itertuples(index=False):
                oof.append({"sample_id":row.sample_id,"speaker_id":row.speaker_id,"session_id":row.session_id,
                            "model":name,"target":target,"y_true":row.y_true,"pred":row.pred})
        ns=len(speakers_ref)
        counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
        for name,(_,mm) in aligned.items():
            boot[name]={}
            for metric in ["overall","between","within"]:
                vals=ccc((counts@mm[metric])/ns); boot[name][metric]=vals; lo,hi=qci(vals)
                metrics.append({"target":target,"model":name,"metric":metric,"ccc":point(mm[metric]),"ci95_low":lo,"ci95_high":hi})
        for metric in ["overall","between","within"]:
            scores={name:point(mm[metric]) for name,(_,mm) in aligned.items()}
            ranks.append({"target":target,"metric":metric,"ranking":" > ".join(sorted(scores,key=scores.get,reverse=True)),**scores})
            for x,y in itertools.combinations(sorted(aligned),2):
                dv=boot[x][metric]-boot[y][metric]; lo,hi=qci(dv)
                deltas.append({"target":target,"metric":metric,"comparison":f"{x}-{y}",
                               "delta_ccc":scores[x]-scores[y],"ci95_low":lo,"ci95_high":hi,
                               "ci_excludes_zero":bool(lo>0 or hi<0)})
        for x,y in itertools.combinations(sorted(aligned),2):
            dv=(boot[x]["between"]-boot[y]["between"])-(boot[x]["within"]-boot[y]["within"])
            lo,hi=qci(dv)
            pt=(point(aligned[x][1]["between"])-point(aligned[y][1]["between"]))-(point(aligned[x][1]["within"])-point(aligned[y][1]["within"]))
            contrasts.append({"target":target,"comparison":f"{x}-{y}","contrast":"between_minus_within",
                              "delta_of_deltas":pt,"ci95_low":lo,"ci95_high":hi,
                              "ci_excludes_zero":bool(lo>0 or hi<0)})
    pd.DataFrame(oof).to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(audit).to_csv(root/"training_audit.csv",index=False)
    pd.DataFrame(metrics).to_csv(root/"component_metrics.csv",index=False)
    pd.DataFrame(deltas).to_csv(root/"pairwise_component_deltas.csv",index=False)
    pd.DataFrame(contrasts).to_csv(root/"pairwise_profile_contrasts.csv",index=False)
    pd.DataFrame(ranks).to_csv(root/"rank_summary.csv",index=False)
    sig=pd.DataFrame(contrasts); sig=sig[sig.ci_excludes_zero]
    summary={"experiment_id":cfg["experiment_id"],"common_samples":len(q),"speakers":int(q.speaker_id.nunique()),
             "sessions":sessions,"fold_policy":"leave-one-session-out","models":list(models),
             "bootstrap_unit":"speaker","bootstrap_reps":reps,
             "profile_contrast_cells":len(contrasts),"significant_profile_contrast_cells":int(len(sig)),
             "significant_profile_contrasts":sig.to_dict(orient="records"),
             "suggested_decision":"pass" if len(sig) else "mixed"}
    (root/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (root/"run_metadata.json").write_text(json.dumps({"experiment_id":cfg["experiment_id"],
        "task":"raw absolute VAD","relative_role":"diagnostic decomposition only",
        "fold_policy":"strict leave-one-session-out","ridge_alpha":alpha,"ridge_solver":solver,
        "bootstrap_seed":cfg["bootstrap_seed"],
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},indent=2)+"\n")
    print(pd.DataFrame(ranks).to_string(index=False))
    print("\nSIGNIFICANT PROFILE CONTRASTS")
    print(sig.to_string(index=False))
if __name__=="__main__": main()
