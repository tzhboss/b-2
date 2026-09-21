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

def stable_seed(*parts):
    return int.from_bytes("|".join(map(str,parts)).encode(),"little")%(2**32)

def speaker_folds(speakers,n_folds,seed):
    s=np.array(sorted(map(str,speakers)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(s)
    return {sp:i%n_folds for i,sp in enumerate(s)}

def equal_speaker_weights(df):
    c=df.speaker_id.astype(str).value_counts()
    w=df.speaker_id.astype(str).map(lambda s:1.0/c[s]).to_numpy(float)
    return w/w.mean()

def weighted_ccc(y,p,w):
    y=np.asarray(y,float); p=np.asarray(p,float); w=np.asarray(w,float); w=w/w.sum()
    my=np.sum(w*y); mp=np.sum(w*p)
    dy=y-my; dp=p-mp
    vy=np.sum(w*dy*dy); vp=np.sum(w*dp*dp); cov=np.sum(w*dy*dp)
    den=vy+vp+(my-mp)**2
    return float(2*cov/den) if den>0 else np.nan

def fit_predict(xtr,xte,ytr,wtr,alpha,solver,tol):
    sc=StandardScaler()
    sc.fit(xtr,sample_weight=wtr)
    model=Ridge(alpha=alpha,solver=solver,tol=tol)
    model.fit(sc.transform(xtr),ytr,sample_weight=wtr)
    return model.predict(sc.transform(xte))

def ccc_from_moments(a):
    my=a[:,0]; ey2=a[:,1]; mp=a[:,2]; ep2=a[:,3]; eyp=a[:,4]
    vy=ey2-my*my; vp=ep2-mp*mp; cov=eyp-my*mp
    den=vy+vp+(my-mp)**2
    out=np.full(len(my),np.nan,float); ok=den>0
    out[ok]=2*cov[ok]/den[ok]
    return out

def per_speaker_moments(g,speakers):
    rows=[]
    for sp in speakers:
        z=g[g.speaker_id==sp]
        y=z.y_true.to_numpy(float)
        pa=z.pred_absolute.to_numpy(float)
        pr=z.pred_relative.to_numpy(float)
        rows.append([y.mean(),np.mean(y*y),pa.mean(),np.mean(pa*pa),np.mean(y*pa),
                     pr.mean(),np.mean(pr*pr),np.mean(y*pr)])
    return np.asarray(rows,float)

def effects_from_counts(counts,stats,lambdas,S):
    w=counts.astype(float)/float(S); out=[]
    for lam in lambdas:
        a=w@stats[lam]
        aa=np.column_stack([a[:,0],a[:,1],a[:,2],a[:,3],a[:,4]])
        rr=np.column_stack([a[:,0],a[:,1],a[:,5],a[:,6],a[:,7]])
        out.append(ccc_from_moments(rr)-ccc_from_moments(aa))
    return np.column_stack(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    # Labels only; avoid reading embedded audio here.
    label_frames=[]
    for f in sorted(glob.glob(cfg["dataset"]["source_glob"])):
        label_frames.append(pd.read_parquet(f,columns=["file","EmoAct","EmoDom"]))
    lab=pd.concat(label_frames,ignore_index=True).dropna().copy()
    lab["sample_id"]=lab.file.astype(str)
    lab["speaker_id"]=lab.file.map(speaker_id)
    lab=lab.rename(columns={"EmoAct":"arousal","EmoDom":"dominance"})

    ids=[]; blocks=[]; inventory=[]
    eroot=Path(cfg["outputs"]["embedding_root"])
    for f in sorted(eroot.glob("*.npz")):
        z=np.load(f)
        ids.extend(z["sample_id"].astype(str).tolist())
        blocks.append(z["embedding"].astype(np.float32))
        inventory.append({"file":f.name,"rows":len(z["sample_id"]),"shape":str(tuple(z["embedding"].shape))})
    emb=np.concatenate(blocks,axis=0)
    em=pd.DataFrame({"sample_id":ids,"row_idx":np.arange(len(ids))})
    d=lab.merge(em,on="sample_id",how="inner",validate="one_to_one")
    if len(d)!=len(lab):
        raise RuntimeError(f"embedding coverage {len(d)}/{len(lab)}")

    lambdas=[float(x) for x in cfg["parameters"]["lambda_values"]]
    targets=list(cfg["parameters"]["targets"])
    layers=list(cfg["parameters"]["layers"])
    speakers=sorted(d.speaker_id.unique())
    fmap=speaker_folds(speakers,int(cfg["parameters"]["outer_folds"]),int(cfg["parameters"]["split_seed"]))
    sf=d.speaker_id.map(fmap).to_numpy()
    oof=[]

    target_parts={}
    for t in targets:
        spm=d.groupby("speaker_id")[t].mean()
        gm=float(d[t].mean())
        target_parts[t]=(d.speaker_id.map(spm).to_numpy(float),gm,d[t].to_numpy(float))

    for lpos,layer in enumerate(layers):
        H=emb[d.row_idx.to_numpy(),lpos,:]
        centers={}
        for sp in speakers:
            idx=np.flatnonzero(d.speaker_id.to_numpy()==sp)
            centers[sp]=np.median(H[idx],axis=0)
        C=np.vstack([centers[sp] for sp in d.speaker_id])
        R=H-C
        for fold in range(int(cfg["parameters"]["outer_folds"])):
            tr=sf!=fold; te=sf==fold
            wtr=equal_speaker_weights(d.loc[tr])
            for lam in lambdas:
                ycols=[]
                for t in targets:
                    spm,gm,raw=target_parts[t]
                    within=raw-spm
                    y=gm+within+lam*(spm-gm)
                    ycols.append(y)
                Y=np.column_stack(ycols)
                pred={}
                solver=str(cfg["parameters"]["ridge_solver"])
                tol=float(cfg["parameters"]["ridge_tol"])
                pred["absolute"]=fit_predict(H[tr],H[te],Y[tr],wtr,float(cfg["parameters"]["ridge_alpha"]),solver,tol)
                pred["relative"]=fit_predict(R[tr],R[te],Y[tr],wtr,float(cfg["parameters"]["ridge_alpha"]),solver,tol)
                test_idx=np.flatnonzero(te)
                for j,t in enumerate(targets):
                    for pos,ri in enumerate(test_idx):
                        oof.append({"sample_id":d.iloc[ri].sample_id,"speaker_id":d.iloc[ri].speaker_id,
                                    "layer":layer,"fold":fold,"lambda":lam,"target":t,
                                    "y_true":float(Y[ri,j]),
                                    "pred_absolute":float(pred["absolute"][pos,j]),
                                    "pred_relative":float(pred["relative"][pos,j]),
                                    "pred_hybrid":np.nan})

    o=pd.DataFrame(oof)
    point=[]; boot=[]
    B=int(cfg["parameters"]["bootstrap_reps"])
    for layer in layers:
      ql=o[o.layer==layer]
      for t in targets:
        q=ql[ql.target==t]
        stats={}
        effects=[]
        for lam in lambdas:
            g=q[q["lambda"]==lam]
            stats[lam]=per_speaker_moments(g,speakers)
            counts=np.ones((1,len(speakers)),dtype=np.int64)
            eff=effects_from_counts(counts,{lam:stats[lam]},[lam],len(speakers))[0,0]
            effects.append(eff)
            point.append({"layer":layer,"target":t,"lambda":lam,"relative_minus_absolute":eff})
        point_slope=float(np.polyfit(lambdas,effects,1)[0])
        rng=np.random.default_rng(stable_seed(layer,t,"speaker_bootstrap"))
        S=len(speakers); pvec=np.full(S,1.0/S)
        bs_s=np.empty(B); bs_l0=np.empty(B)
        x=np.asarray(lambdas,float); xc=x-x.mean(); den=float(xc@xc)
        off=0
        while off<B:
            n=min(250,B-off)
            counts=rng.multinomial(S,pvec,size=n)
            eff=effects_from_counts(counts,stats,lambdas,S)
            bs_s[off:off+n]=(eff@xc)/den
            bs_l0[off:off+n]=eff[:,0]
            off+=n
        slo,shi=np.quantile(bs_s,[.025,.975]); elo,ehi=np.quantile(bs_l0,[.025,.975])
        boot.append({"layer":layer,"target":t,"slope_mean":point_slope,
                     "slope_ci95_low":float(slo),"slope_ci95_high":float(shi),
                     "lambda0_effect":effects[0],"lambda0_ci95_low":float(elo),"lambda0_ci95_high":float(ehi),
                     "speakers":S})

    default=pd.read_csv(cfg["inputs"]["default_results"])
    comp=pd.DataFrame(boot).merge(default[["layer","target","slope_mean","lambda0_effect"]],
                                  on=["layer","target"],suffixes=("_lsqr","_default"))
    comp["slope_abs_diff"]=(comp.slope_mean_lsqr-comp.slope_mean_default).abs()
    comp["lambda0_abs_diff"]=(comp.lambda0_effect_lsqr-comp.lambda0_effect_default).abs()

    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(inventory).to_csv(root/"embedding_inventory.csv",index=False)
    o.to_parquet(root/"oof_predictions.parquet",index=False)
    pd.DataFrame(point).to_csv(root/"point_effects.csv",index=False)
    pd.DataFrame(boot).to_csv(root/"cluster_bootstrap_slopes.csv",index=False)
    comp.to_csv(root/"solver_comparison.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],"layers":layers,"rows":len(d),"speakers":len(speakers),
      "split_seed":cfg["parameters"]["split_seed"],"bootstrap_unit":"speaker","ridge_solver":cfg["parameters"]["ridge_solver"],"ridge_tol":cfg["parameters"]["ridge_tol"]
    },indent=2)+"\n")
    print(pd.DataFrame(boot).to_string(index=False))

if __name__=="__main__":
    main()
