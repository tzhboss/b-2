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

def ccc_m(m):
    m=np.asarray(m,float); my,mp,my2,mp2,myp=[m[...,i] for i in range(5)]
    vy=np.maximum(my2-my*my,0); vp=np.maximum(mp2-mp*mp,0); den=vy+vp+(my-mp)**2
    return np.divide(2*(myp-my*mp),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)

def ccc_raw(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.array([y.mean(),p.mean(),np.mean(y*y),np.mean(p*p),np.mean(y*p)])
    return float(ccc_m(m))

def moments(z,ycol,pcol):
    x=z[["speaker_id",ycol,pcol]].copy()
    x["y2"]=x[ycol]**2; x["p2"]=x[pcol]**2; x["yp"]=x[ycol]*x[pcol]
    g=x.groupby("speaker_id",sort=True).agg(
      my=(ycol,"mean"),mp=(pcol,"mean"),my2=("y2","mean"),mp2=("p2","mean"),myp=("yp","mean"))
    o=g[["my","mp","my2","mp2","myp"]].to_numpy(float)
    b=np.column_stack([g.my,g.mp,g.my**2,g.mp**2,g.my*g.mp])
    w=np.column_stack([np.zeros(len(g)),np.zeros(len(g)),np.maximum(g.my2-g.my**2,0),np.maximum(g.mp2-g.mp**2,0),g.myp-g.my*g.mp])
    return list(g.index.astype(str)),{"overall":o,"between":b,"within":w}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); root=Path(cfg["output_root"])
    files=sorted(glob.glob(str(root/"seed-*-Ses*"/"predictions.parquet")))
    expected=len(cfg["seeds"])*len(cfg["sessions"])
    if len(files)!=expected: raise RuntimeError(f"expected {expected} shards, got {len(files)}")
    O=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    key=["sample_id","seed"]
    if O.duplicated(key).any(): raise RuntimeError("duplicate OOF sample/seed")
    if not np.isfinite(O.select_dtypes(include=[np.number]).to_numpy()).all(): raise RuntimeError("nonfinite OOF")
    O.to_parquet(root/"oof_predictions.parquet",index=False)

    metric_rows=[]; cache={}
    for seed in [int(x) for x in cfg["seeds"]]:
        z=O[O.seed.eq(seed)].copy()
        for t in TARGETS:
            ycol=f"y_{t}"
            for v,prefix in VARIANTS.items():
                pcol=f"{prefix}_{t}"
                sp,mm=moments(z,ycol,pcol); cache[(seed,t,v)]=(sp,mm)
                raw=ccc_raw(z[ycol],z[pcol])
                for metric in ["overall","between","within"]:
                    metric_rows.append({
                      "seed":seed,"target":t,"variant":v,"metric":metric,
                      "ccc":float(ccc_m(mm[metric].mean(0))),
                      "raw_overall_ccc":raw if metric=="overall" else np.nan
                    })
    M=pd.DataFrame(metric_rows); M.to_csv(root/"component_metrics_by_seed.csv",index=False)

    reps=int(cfg.get("bootstrap_reps",3000)); paired=[]
    for t in TARGETS:
        for metric in ["overall","between","within"]:
            seed_pts=[]; boots=[]; positive=0
            for seed in [int(x) for x in cfg["seeds"]]:
                sp0,mm0=cache[(seed,t,"A_standard")]; sp1,mm1=cache[(seed,t,"C_within_aux")]
                if sp0!=sp1: raise RuntimeError("speaker ordering mismatch")
                ns=len(sp0)
                pt=float(ccc_m(mm1[metric].mean(0))-ccc_m(mm0[metric].mean(0)))
                seed_pts.append(pt); positive+=int(pt>0)
                rng=np.random.default_rng(stable_seed(cfg.get("bootstrap_seed",20261003),seed,t,metric))
                counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
                boots.append(ccc_m((counts@mm1[metric])/ns)-ccc_m((counts@mm0[metric])/ns))
            b=np.nanmean(np.stack(boots),axis=0); lo,hi=np.nanquantile(b,[.025,.975])
            paired.append({
              "target":t,"variant":"C_within_aux","reference":"A_standard","metric":metric,
              "delta_ccc_mean_across_seeds":float(np.mean(seed_pts)),
              "ci95_low":float(lo),"ci95_high":float(hi),
              "ci_excludes_zero":bool(lo>0 or hi<0),
              "positive_seeds":positive,"n_seeds":len(seed_pts)
            })
    P=pd.DataFrame(paired); P.to_csv(root/"paired_component_deltas.csv",index=False)

    rawrows=[]
    for seed in [int(x) for x in cfg["seeds"]]:
        z=O[O.seed.eq(seed)]
        for t in TARGETS:
            y=f"y_{t}"
            a=ccc_raw(z[y],z[f"A_{t}"]); c=ccc_raw(z[y],z[f"C_{t}"])
            rawrows.append({"seed":seed,"target":t,"A_raw_ccc":a,"C_raw_ccc":c,"delta_raw_ccc":c-a})
    R=pd.DataFrame(rawrows); R.to_csv(root/"standard_raw_ccc_by_seed.csv",index=False)
    RS=R.groupby("target",as_index=False).agg(
      A_raw_ccc_mean=("A_raw_ccc","mean"),C_raw_ccc_mean=("C_raw_ccc","mean"),
      delta_raw_ccc_mean=("delta_raw_ccc","mean"),positive_seeds=("delta_raw_ccc",lambda x:int((x>0).sum())))
    RS.to_csv(root/"standard_raw_ccc_summary.csv",index=False)

    audits=[json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(root/"seed-*-Ses*"/"audit.json")))]
    if len(audits)!=expected: raise RuntimeError("missing shard audits")
    audit={
      "experiment_id":cfg["experiment_id"],"validity":"valid","task":"standard absolute IEMOCAP VAD",
      "training":"fine-tuned WavLM encoder with feature-extractor CNN frozen",
      "outer_split":"5-session LOSO","seeds":[int(x) for x in cfg["seeds"]],
      "shards":len(files),"samples_per_seed":int(len(O)/len(cfg["seeds"])),
      "speakers":int(O.speaker_id.nunique()),"duplicate_oof_keys":int(O.duplicated(key).sum()),
      "test_speaker_history_or_mean_used_as_input":False
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("RAW STANDARD CCC"); print(RS.to_string(index=False))
    print("\nPAIRED COMPONENT DELTAS"); print(P.to_string(index=False))
    print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
