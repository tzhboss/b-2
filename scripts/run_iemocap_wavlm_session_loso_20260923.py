#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def speaker_id(x):
    m=re.match(r'(Ses\d\d)[FM]_.+_([FM])\d+\.wav$',str(x))
    if not m: raise ValueError(x)
    return m.group(1)+'_'+m.group(2)

def session_id(x):
    m=re.match(r'^(Ses\d\d)',str(x))
    if not m: raise ValueError(x)
    return m.group(1)

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

def fit_predict(xtr,xte,ytr,wtr,alpha):
    sc=StandardScaler(); sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha)
    model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())

    frames=[]
    for f in sorted(glob.glob(cfg["dataset"]["source_glob"])):
        frames.append(pd.read_parquet(f,columns=["file","EmoAct","EmoDom"]))
    lab=pd.concat(frames,ignore_index=True).dropna().copy()
    lab["sample_id"]=lab.file.astype(str)
    lab["speaker_id"]=lab.file.map(speaker_id)
    lab["session_id"]=lab.file.map(session_id)
    lab=lab.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})

    ids=[]; blocks=[]
    eroot=Path(cfg["outputs"]["embedding_root"])
    for f in sorted(eroot.glob("*.npz")):
        z=np.load(f)
        ids.extend(z["sample_id"].astype(str).tolist())
        blocks.append(z["embedding"].astype(np.float32))
    emb=np.concatenate(blocks,axis=0)
    em=pd.DataFrame({"sample_id":ids,"row_idx":np.arange(len(ids))})
    d=lab.merge(em,on="sample_id",how="inner",validate="one_to_one")
    if len(d)!=len(lab): raise RuntimeError(f"embedding coverage {len(d)}/{len(lab)}")

    layers=[int(x) for x in cfg["parameters"]["layers"]]
    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    sessions=sorted(d.session_id.unique())
    if sessions!=["Ses01","Ses02","Ses03","Ses04","Ses05"]:
        raise RuntimeError(f"unexpected sessions {sessions}")

    target_parts={}
    for t in targets:
        spm=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        target_parts[t]=(d.speaker_id.map(spm).to_numpy(float),gm,d[t].to_numpy(float))

    curves=[]; oof=[]
    for lpos,layer in enumerate(layers):
        H=emb[d.row_idx.to_numpy(),lpos,:]
        speakers=sorted(d.speaker_id.unique())
        centers={}
        for sp in speakers:
            idx=np.flatnonzero(d.speaker_id.to_numpy()==sp)
            centers[sp]=np.median(H[idx],axis=0)
        C=np.vstack([centers[sp] for sp in d.speaker_id])
        R=H-C

        for held in sessions:
            te=d.session_id.eq(held).to_numpy(); tr=~te
            train=d.loc[tr]; test=d.loc[te]
            if set(train.speaker_id)&set(test.speaker_id):
                raise RuntimeError(f"speaker leakage for {held}")
            wtr=equal_speaker_weights(train); wte=equal_speaker_weights(test)
            for lam in lambdas:
                Y=np.column_stack([
                    target_parts[t][1] + (target_parts[t][2]-target_parts[t][0]) + lam*(target_parts[t][0]-target_parts[t][1])
                    for t in targets
                ])
                pa=fit_predict(H[tr],H[te],Y[tr],wtr,float(cfg["parameters"]["ridge_alpha"]))
                pr=fit_predict(R[tr],R[te],Y[tr],wtr,float(cfg["parameters"]["ridge_alpha"]))
                test_idx=np.flatnonzero(te)
                for j,t in enumerate(targets):
                    ca=weighted_ccc(Y[te,j],pa[:,j],wte)
                    cr=weighted_ccc(Y[te,j],pr[:,j],wte)
                    curves.append({"held_session":held,"layer":layer,"target":t,"lambda":lam,
                                   "ccc_absolute":ca,"ccc_relative":cr,"relative_minus_absolute":cr-ca,
                                   "test_speakers":test.speaker_id.nunique(),"test_rows":len(test)})
                    for pos,ri in enumerate(test_idx):
                        oof.append({"sample_id":d.iloc[ri].sample_id,"speaker_id":d.iloc[ri].speaker_id,
                                    "session_id":held,"layer":layer,"target":t,"lambda":lam,
                                    "y_true":float(Y[ri,j]),"pred_absolute":float(pa[pos,j]),
                                    "pred_relative":float(pr[pos,j])})

    c=pd.DataFrame(curves)
    slopes=[]
    for (held,layer,target),g in c.groupby(["held_session","layer","target"],sort=True):
        g=g.sort_values("lambda")
        slope=float(np.polyfit(g["lambda"].to_numpy(float),g.relative_minus_absolute.to_numpy(float),1)[0])
        slopes.append({"held_session":held,"layer":int(layer),"target":target,"slope":slope,
                       "lambda0_effect":float(g.iloc[0].relative_minus_absolute),
                       "lambda1_effect":float(g.iloc[-1].relative_minus_absolute)})
    s=pd.DataFrame(slopes)

    summary=[]
    delete_rows=[]
    for (layer,target),g in s.groupby(["layer","target"],sort=True):
        vals=g.set_index("held_session").slope
        loo=[]
        for held in sessions:
            mean4=float(vals.drop(held).mean())
            loo.append(mean4)
            delete_rows.append({"layer":int(layer),"target":target,"deleted_session":held,
                                "mean_slope_remaining4":mean4})
        summary.append({"layer":int(layer),"target":target,"mean_session_slope":float(vals.mean()),
                        "negative_sessions":int((vals<0).sum()),"n_sessions":len(vals),
                        "min_session_slope":float(vals.min()),"max_session_slope":float(vals.max()),
                        "delete1_mean_min":float(min(loo)),"delete1_mean_max":float(max(loo))})

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    c.to_csv(root/"session_effect_curves.csv",index=False)
    s.to_csv(root/"session_slopes.csv",index=False)
    pd.DataFrame(summary).to_csv(root/"session_summary.csv",index=False)
    pd.DataFrame(delete_rows).to_csv(root/"delete_one_session_sensitivity.csv",index=False)
    pd.DataFrame(oof).to_parquet(root/"oof_predictions.parquet",index=False)
    meta={"experiment_id":cfg["experiment_id"],"sessions":sessions,"layers":layers,
          "rows":len(d),"speakers":d.speaker_id.nunique(),
          "split_protocol":"leave-one-session-out; both speakers of held session only in test",
          "center_protocol":"oracle unlabeled per-speaker median WavLM embedding",
          "inference_note":"five-session protocol robustness; no asymptotic significance claim"}
    (root/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nPER SESSION")
    print(s.to_string(index=False))

if __name__=="__main__":
    main()
