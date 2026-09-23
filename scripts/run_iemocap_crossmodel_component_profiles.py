#!/usr/bin/env python3
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def ccc_from_moments(m):
    m=np.asarray(m,float)
    my,mp,my2,mp2,myp=[m[...,i] for i in range(5)]
    vy=np.maximum(my2-my*my,0.0); vp=np.maximum(mp2-mp*mp,0.0); cov=myp-my*mp
    den=vy+vp+(my-mp)**2
    return np.divide(2*cov,den,out=np.full_like(den,np.nan,dtype=float),where=den>0)

def moments(df):
    x=df.copy()
    x["y2"]=x.y_true*x.y_true; x["p2"]=x.pred*x.pred; x["yp"]=x.y_true*x.pred
    g=x.groupby("speaker_id",sort=True).agg(
        my=("y_true","mean"),mp=("pred","mean"),my2=("y2","mean"),mp2=("p2","mean"),myp=("yp","mean"))
    overall=g[["my","mp","my2","mp2","myp"]].to_numpy(float)
    between=np.column_stack([g.my,g.mp,g.my**2,g.mp**2,g.my*g.mp]).astype(float)
    within=np.column_stack([
        np.zeros(len(g)),np.zeros(len(g)),
        np.maximum(g.my2-g.my**2,0),np.maximum(g.mp2-g.mp**2,0),g.myp-g.my*g.mp]).astype(float)
    return list(g.index.astype(str)),{"overall":overall,"between":between,"within":within}

def point(m):
    return float(ccc_from_moments(m.mean(0)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    lam=float(cfg.get("lambda_filter",1.0)); reps=int(cfg.get("bootstrap_reps",10000)); seed=int(cfg.get("bootstrap_seed",20260923))
    root=Path(cfg["output_root"]); root.mkdir(parents=True,exist_ok=True)

    pr=pd.read_parquet(cfg["pitch_rate_oof"])
    pr=pr[(pr.dataset=="iemocap") & np.isclose(pr["lambda"].astype(float),lam)].copy()
    w=pd.read_parquet(cfg["wavlm_oof"])
    w=w[np.isclose(w["lambda"].astype(float),lam)].copy()

    tables={}
    for fam,name in [("linear_ridge","pitch_rate_linear"),("quadratic_ridge","pitch_rate_quadratic")]:
        z=pr[pr.model_family==fam][["sample_id","speaker_id","target","y_true","pred_absolute"]].rename(columns={"pred_absolute":"pred"})
        z["model"]=name; tables[name]=z
    for layer in [12,24]:
        name=f"wavlm_layer{layer}"
        z=w[w.layer==layer][["sample_id","speaker_id","target","y_true","pred_absolute"]].rename(columns={"pred_absolute":"pred"})
        z["model"]=name; tables[name]=z

    audit=[]; point_rows=[]; pair_rows=[]; rank_rows=[]; reversal_rows=[]
    rng=np.random.default_rng(seed)

    for target in ["arousal","dominance"]:
        ts={k:v[v.target==target].copy() for k,v in tables.items()}
        sample_sets={k:set(v.sample_id.astype(str)) for k,v in ts.items()}
        common=set.intersection(*sample_sets.values()); union=set.union(*sample_sets.values())
        audit.append({"target":target,"common_samples":len(common),"union_samples":len(union),
                      **{f"n_{k}":len(v) for k,v in ts.items()}})
        if not common: raise RuntimeError(f"no common samples for {target}")

        aligned={}; truth_ref=None
        for name,z in ts.items():
            z=z[z.sample_id.astype(str).isin(common)].copy().sort_values("sample_id")
            if z.sample_id.duplicated().any(): raise RuntimeError(f"duplicates {name}/{target}")
            key=z[["sample_id","speaker_id","y_true"]].reset_index(drop=True)
            if truth_ref is None:
                truth_ref=key
            else:
                if not np.array_equal(key.sample_id.to_numpy(),truth_ref.sample_id.to_numpy()):
                    raise RuntimeError("sample alignment failed")
                if not np.array_equal(key.speaker_id.astype(str).to_numpy(),truth_ref.speaker_id.astype(str).to_numpy()):
                    raise RuntimeError("speaker mismatch")
                if not np.allclose(key.y_true.to_numpy(float),truth_ref.y_true.to_numpy(float),rtol=0,atol=1e-8):
                    raise RuntimeError("target mismatch")
            sp,mm=moments(z); aligned[name]=(sp,mm)

        speakers=aligned[next(iter(aligned))][0]
        if any(v[0]!=speakers for v in aligned.values()): raise RuntimeError("speaker-set mismatch")
        ns=len(speakers)
        counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)
        boot={}
        for name,(sp,mm) in aligned.items():
            boot[name]={}
            for metric in ["overall","between","within"]:
                vals=ccc_from_moments((counts@mm[metric])/ns); boot[name][metric]=vals
                lo,hi=np.nanquantile(vals,[.025,.975])
                point_rows.append({"target":target,"model":name,"metric":metric,"ccc":point(mm[metric]),
                                   "ci95_low":float(lo),"ci95_high":float(hi),"n_speakers":ns,"n_samples":len(common)})

        for metric in ["overall","between","within"]:
            scores={n:point(v[1][metric]) for n,v in aligned.items()}
            ordered=sorted(scores,key=scores.get,reverse=True)
            rank_rows.append({"target":target,"metric":metric,"ranking":" > ".join(ordered),
                              **{n:scores[n] for n in sorted(scores)}})
            for a,b in itertools.combinations(sorted(aligned),2):
                vals=boot[a][metric]-boot[b][metric]
                lo,hi=np.nanquantile(vals,[.025,.975])
                pair_rows.append({"target":target,"metric":metric,"comparison":f"{a}-{b}",
                                  "delta_ccc":scores[a]-scores[b],"ci95_low":float(lo),"ci95_high":float(hi)})

        overall={n:point(v[1]["overall"]) for n,v in aligned.items()}
        within={n:point(v[1]["within"]) for n,v in aligned.items()}
        for a,b in itertools.combinations(sorted(aligned),2):
            so=np.sign(overall[a]-overall[b]); sw=np.sign(within[a]-within[b])
            reversal_rows.append({"target":target,"pair":f"{a} vs {b}",
                                  "overall_delta":overall[a]-overall[b],"within_delta":within[a]-within[b],
                                  "ordering_reversal":bool(so!=0 and sw!=0 and so!=sw)})

    pd.DataFrame(audit).to_csv(root/"sample_alignment.csv",index=False)
    pd.DataFrame(point_rows).to_csv(root/"component_metrics.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(root/"pairwise_deltas.csv",index=False)
    ranks=pd.DataFrame(rank_rows); ranks.to_csv(root/"rank_summary.csv",index=False)
    rev=pd.DataFrame(reversal_rows); rev.to_csv(root/"ordering_reversals.csv",index=False)
    meta={"experiment_id":cfg["experiment_id"],"lambda_filter":lam,"bootstrap_unit":"speaker",
          "bootstrap_reps":reps,"bootstrap_seed":seed,"prediction_condition":"absolute",
          "note":"standard absolute VAD; Within is diagnostic centering of OOF truth/prediction"}
    (root/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("ALIGNMENT"); print(pd.DataFrame(audit).to_string(index=False))
    print("\nMETRICS"); print(pd.DataFrame(point_rows).to_string(index=False))
    print("\nRANKS"); print(ranks.to_string(index=False))
    print("\nREVERSALS"); print(rev.to_string(index=False))

if __name__=="__main__":
    main()
