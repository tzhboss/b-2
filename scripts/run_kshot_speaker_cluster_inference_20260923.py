#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

REPS={
    "absolute":"pred_absolute",
    "kshot_hybrid":"pred_kshot_hybrid",
    "marginal_oracle_hybrid":"pred_marginal_oracle_hybrid",
}

def stable_seed(*parts):
    payload="|".join(map(str,parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")%(2**32)

def ccc_from_moments(a):
    a=np.asarray(a,float)
    my,mp,my2,mp2,myp=[a[...,i] for i in range(5)]
    vy=np.maximum(my2-my*my,0.0); vp=np.maximum(mp2-mp*mp,0.0)
    cov=myp-my*mp; den=vy+vp+(my-mp)**2
    return np.divide(2*cov,den,out=np.full_like(den,np.nan,dtype=float),where=den>0)

def speaker_moments(df,pred_col):
    x=df[["speaker_id","y_true",pred_col]].rename(columns={pred_col:"pred"}).copy()
    x["y2"]=x.y_true*x.y_true; x["p2"]=x.pred*x.pred; x["yp"]=x.y_true*x.pred
    g=x.groupby("speaker_id",sort=True).agg(
        my=("y_true","mean"),mp=("pred","mean"),
        my2=("y2","mean"),mp2=("p2","mean"),myp=("yp","mean"))
    return list(g.index.astype(str)),g[["my","mp","my2","mp2","myp"]].to_numpy(float)

def point_ccc(m):
    return float(ccc_from_moments(m.mean(axis=0)))

def bootstrap_contrast(moment_dict,seeds,k,target,a,b,reps,bootstrap_seed):
    speakers=moment_dict[(seeds[0],k,target,a)][0]
    S=len(speakers)
    rng=np.random.default_rng(stable_seed(bootstrap_seed,k,target,a,b))
    counts=rng.multinomial(S,np.full(S,1.0/S),size=reps).astype(float)
    boot_seed=[]
    point_seed=[]
    for seed in seeds:
        sa,ma=moment_dict[(seed,k,target,a)]
        sb,mb=moment_dict[(seed,k,target,b)]
        if sa!=speakers or sb!=speakers:
            raise RuntimeError("speaker order mismatch")
        ca=ccc_from_moments((counts@ma)/S)
        cb=ccc_from_moments((counts@mb)/S)
        boot_seed.append(ca-cb)
        point_seed.append(point_ccc(ma)-point_ccc(mb))
    boot=np.mean(np.vstack(boot_seed),axis=0)
    lo,hi=np.nanquantile(boot,[.025,.975])
    return float(np.mean(point_seed)),float(lo),float(hi),S

def bootstrap_k_delta(moment_dict,seeds,khi,klo,target,reps,bootstrap_seed):
    speakers=moment_dict[(seeds[0],khi,target,"kshot_hybrid")][0]
    S=len(speakers)
    rng=np.random.default_rng(stable_seed(bootstrap_seed,target,khi,klo,"kdelta"))
    counts=rng.multinomial(S,np.full(S,1.0/S),size=reps).astype(float)
    boots=[]; pts=[]
    for seed in seeds:
        shi,mhi=moment_dict[(seed,khi,target,"kshot_hybrid")]
        slo,mlo=moment_dict[(seed,klo,target,"kshot_hybrid")]
        if shi!=speakers or slo!=speakers:
            raise RuntimeError("speaker order mismatch")
        boots.append(ccc_from_moments((counts@mhi)/S)-ccc_from_moments((counts@mlo)/S))
        pts.append(point_ccc(mhi)-point_ccc(mlo))
    boot=np.mean(np.vstack(boots),axis=0)
    lo,hi=np.nanquantile(boot,[.025,.975])
    return float(np.mean(pts)),float(lo),float(hi),S

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["outputs"]["artifact_root"])
    seeds=[int(x) for x in cfg["seed"]]
    ks=[int(x) for x in cfg["parameters"]["k_values"]]
    targets=list(cfg["parameters"]["targets"])
    B=int(cfg["parameters"]["bootstrap_reps"])
    bseed=int(cfg["parameters"].get("bootstrap_seed",20260923))

    moment_dict={}; summary=[]; inventory=[]
    speakers_ref=None
    for seed in seeds:
        for k in ks:
            f=root/f"paired_oof_seed_{seed}_K{k}.parquet"
            if not f.exists(): raise FileNotFoundError(f)
            d=pd.read_parquet(f)
            inventory.append({"seed":seed,"K":k,"rows":len(d),"speakers":d.speaker_id.nunique(),
                              "samples":d.sample_id.nunique()})
            if d.duplicated(["sample_id","target"]).any():
                raise RuntimeError(f"duplicate OOF rows in {f}")
            for target in targets:
                q=d[d.target.eq(target)]
                for rep,col in REPS.items():
                    speakers,m=speaker_moments(q,col)
                    if speakers_ref is None: speakers_ref=speakers
                    if speakers!=speakers_ref: raise RuntimeError("speaker set/order differs across seed/K")
                    moment_dict[(seed,k,target,rep)]=(speakers,m)
                    summary.append({"seed":seed,"K":k,"target":target,"representation":rep,
                                    "ccc":point_ccc(m),"n_speakers":len(speakers)})

    comparisons=[
        ("kshot_hybrid","absolute"),
        ("kshot_hybrid","marginal_oracle_hybrid"),
        ("marginal_oracle_hybrid","absolute"),
    ]
    deltas=[]
    for k in ks:
        for target in targets:
            for a,b in comparisons:
                pt,lo,hi,S=bootstrap_contrast(moment_dict,seeds,k,target,a,b,B,bseed)
                deltas.append({"K":k,"target":target,"comparison":f"{a}-{b}",
                               "delta_ccc_mean_across_seeds":pt,"ci95_low":lo,"ci95_high":hi,
                               "n_speakers":S,"n_modeling_seeds":len(seeds),"bootstrap_reps":B})

    kd=[]
    for target in targets:
        for khi,klo in [(2,1),(5,2),(10,5),(20,10),(50,20)]:
            pt,lo,hi,S=bootstrap_k_delta(moment_dict,seeds,khi,klo,target,B,bseed)
            kd.append({"target":target,"comparison":f"K{khi}-K{klo}",
                       "delta_ccc_mean_across_seeds":pt,"ci95_low":lo,"ci95_high":hi,
                       "n_speakers":S,"n_modeling_seeds":len(seeds),"bootstrap_reps":B})

    pd.DataFrame(inventory).to_csv(root/"oof_inventory.csv",index=False)
    pd.DataFrame(summary).to_csv(root/"summary_by_seed.csv",index=False)
    pd.DataFrame(deltas).to_csv(root/"speaker_cluster_deltas.csv",index=False)
    pd.DataFrame(kd).to_csv(root/"speaker_cluster_k_deltas.csv",index=False)
    meta={
        "experiment_id":cfg["experiment_id"],
        "independent_unit":"speaker",
        "bootstrap_reps":B,
        "bootstrap_seed":bseed,
        "seed_handling":"same speaker bootstrap counts applied to each modeling seed; CCC contrasts computed within seed then averaged across seeds",
        "point_estimand":"mean of seed-specific full-OOF CCC contrasts",
        "n_speakers":len(speakers_ref),
        "modeling_seeds":seeds,
    }
    (root/"speaker_cluster_inference_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("DELTAS")
    print(pd.DataFrame(deltas).to_string(index=False))
    print("\nK DELTAS")
    print(pd.DataFrame(kd).to_string(index=False))

if __name__=="__main__":
    main()
