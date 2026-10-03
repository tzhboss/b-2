#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

TARGETS=["arousal","valence","dominance"]
VARIANTS={"A_standard":"A","C_within_aux":"C"}

def stable_seed(*parts):
    h=hashlib.sha256("::".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:4],"little")

def ccc_from_m(m):
    my,mp,y2,p2,yp=[np.asarray(x,float) for x in np.moveaxis(np.asarray(m,float),-1,0)]
    vy=y2-my*my; vp=p2-mp*mp; cov=yp-my*mp
    den=vy+vp+(my-mp)**2
    return np.where(den>0,2*cov/den,np.nan)

def speaker_moments(z,ycol,pcol,kind):
    rows=[]
    for sp,g in z.groupby("speaker_id",sort=True):
        y=g[ycol].to_numpy(float); p=g[pcol].to_numpy(float)
        if kind=="between":
            yy=np.array([y.mean()]); pp=np.array([p.mean()])
        elif kind=="within":
            yy=y-y.mean(); pp=p-p.mean()
        elif kind=="overall":
            yy=y; pp=p
        else: raise ValueError(kind)
        rows.append([str(sp),yy.mean(),pp.mean(),np.mean(yy*yy),np.mean(pp*pp),np.mean(yy*pp)])
    return pd.DataFrame(rows,columns=["speaker_id","my","mp","y2","p2","yp"])

def unweighted_ccc(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.array([y.mean(),p.mean(),np.mean(y*y),np.mean(p*p),np.mean(y*p)])
    return float(ccc_from_m(m))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); root=Path(cfg["output_root"])
    files=sorted(glob.glob(str(root/"seed-*-fold-*"/"predictions.parquet")))
    expected=len(cfg["seeds"])*int(cfg["outer_folds"])
    if len(files)!=expected: raise RuntimeError(f"expected {expected} shards, got {len(files)}")
    O=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    key=["sample_id","seed"]
    if O.duplicated(key).any(): raise RuntimeError("duplicate OOF sample/seed")
    if not np.isfinite(O.select_dtypes(include=[np.number]).to_numpy()).all(): raise RuntimeError("nonfinite OOF")
    O.to_parquet(root/"oof_predictions.parquet",index=False)

    metric_rows=[]; raw_rows=[]; moment_cache={}
    for seed in [int(x) for x in cfg["seeds"]]:
        zs=O[O.seed.eq(seed)].copy()
        for t in TARGETS:
            ycol=f"y_{t}"
            raw={}
            for v,prefix in VARIANTS.items():
                pcol=f"{prefix}_{t}"
                uw=unweighted_ccc(zs[ycol],zs[pcol]); raw[v]=uw
                for kind in ["overall","between","within"]:
                    sm=speaker_moments(zs,ycol,pcol,kind)
                    moment_cache[(seed,t,v,kind)]=sm
                    pt=float(ccc_from_m(sm[["my","mp","y2","p2","yp"]].to_numpy().mean(0)))
                    metric_rows.append({"seed":seed,"target":t,"variant":v,"metric":kind,
                                        "speaker_balanced_ccc":pt,"unweighted_overall_ccc":uw if kind=="overall" else np.nan})
            raw_rows.append({"seed":seed,"target":t,
                             "A_raw_ccc":raw["A_standard"],"C_raw_ccc":raw["C_within_aux"],
                             "delta_raw_ccc":raw["C_within_aux"]-raw["A_standard"]})
    M=pd.DataFrame(metric_rows); M.to_csv(root/"component_metrics_by_seed.csv",index=False)
    R=pd.DataFrame(raw_rows); R.to_csv(root/"standard_raw_ccc_by_seed.csv",index=False)
    RS=R.groupby("target",as_index=False).agg(
        A_raw_ccc_mean=("A_raw_ccc","mean"),
        C_raw_ccc_mean=("C_raw_ccc","mean"),
        delta_raw_ccc_mean=("delta_raw_ccc","mean"),
        positive_seeds=("delta_raw_ccc",lambda x:int((x>0).sum())))
    RS.to_csv(root/"standard_raw_ccc_summary.csv",index=False)

    reps=int(cfg.get("bootstrap_reps",3000)); ns=int(O.speaker_id.nunique())
    paired=[]
    for t in TARGETS:
        for kind in ["overall","between","within"]:
            seed_pts=[]; seed_boot=[]; positive=0
            rng=np.random.default_rng(stable_seed(cfg.get("bootstrap_seed",20261003),t,kind))
            counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
            for seed in [int(x) for x in cfg["seeds"]]:
                a=moment_cache[(seed,t,"A_standard",kind)]
                c=moment_cache[(seed,t,"C_within_aux",kind)]
                if a.speaker_id.tolist()!=c.speaker_id.tolist(): raise RuntimeError("speaker order mismatch")
                am=a[["my","mp","y2","p2","yp"]].to_numpy(); cm=c[["my","mp","y2","p2","yp"]].to_numpy()
                apt=float(ccc_from_m(am.mean(0))); cpt=float(ccc_from_m(cm.mean(0))); dv=cpt-apt
                seed_pts.append(dv); positive+=int(dv>0)
                ab=ccc_from_m((counts@am)/ns); cb=ccc_from_m((counts@cm)/ns)
                seed_boot.append(cb-ab)
            b=np.nanmean(np.stack(seed_boot),axis=0)
            lo,hi=np.nanquantile(b,[.025,.975])
            paired.append({"target":t,"variant":"C_within_aux","reference":"A_standard","metric":kind,
                           "delta_ccc_mean_across_seeds":float(np.mean(seed_pts)),
                           "ci95_low":float(lo),"ci95_high":float(hi),
                           "ci_excludes_zero":bool(lo>0 or hi<0),"positive_seeds":positive,"n_seeds":len(seed_pts)})
    P=pd.DataFrame(paired); P.to_csv(root/"paired_component_deltas.csv",index=False)
    audit={
      "experiment_id":cfg["experiment_id"],"validity":"valid","samples_per_seed":int(len(O)/len(cfg["seeds"])),
      "speakers":int(O.speaker_id.nunique()),"outer_split":"5 speaker-disjoint folds",
      "seeds":[int(x) for x in cfg["seeds"]],"wavlm_layer":int(cfg["wavlm_layer"]),
      "backbone_frozen":True,"test_speaker_history_or_mean_used_as_input":False,
      "shards":len(files),"duplicate_oof_keys":int(O.duplicated(key).sum())
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("RAW STANDARD CCC"); print(RS.to_string(index=False))
    print("\nPAIRED COMPONENT DELTAS"); print(P.to_string(index=False))
    print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
