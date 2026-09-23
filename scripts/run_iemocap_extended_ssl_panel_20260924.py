#!/usr/bin/env python3
from __future__ import annotations
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np,pandas as pd,yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_iemocap_paired_crossmodel_component_attribution_20260924 import ccc,moments,point,qci

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg["panel_output_root"]); root.mkdir(parents=True,exist_ok=True)
    for p in root.glob("panel_*"):
        raise RuntimeError(f"refusing to overwrite existing panel output {p}")
    lam=float(cfg["lambda_filter"]); reps=int(cfg["bootstrap_reps"]); seed=int(cfg["bootstrap_seed"])
    pr=pd.read_parquet(cfg["pitch_rate_oof"]); pr=pr[(pr.dataset=="iemocap")&np.isclose(pr["lambda"].astype(float),lam)]
    w=pd.read_parquet(cfg["wavlm_oof"]); w=w[np.isclose(w["lambda"].astype(float),lam)]
    tables={}
    for fam,name in [("linear_ridge","pitch_rate_linear"),("quadratic_ridge","pitch_rate_quadratic")]:
        tables[name]=pr[pr.model_family==fam][["sample_id","speaker_id","target","y_true","pred_absolute"]].rename(columns={"pred_absolute":"pred"})
    for layer in [12,24]:
        tables[f"wavlm_layer{layer}"]=w[w.layer==layer][["sample_id","speaker_id","target","y_true","pred_absolute"]].rename(columns={"pred_absolute":"pred"})
    ez=[]
    for t,p in cfg["emotion2vec_oof"].items(): ez.append(pd.read_parquet(p)[["sample_id","speaker_id","target","y_true","pred"]])
    tables["emotion2vec_plus_large"]=pd.concat(ez,ignore_index=True)
    for name in cfg["models"]:
        zz=[]
        for t in cfg["targets"]: zz.append(pd.read_parquet(root/f"oof_{name}_{t}.parquet")[["sample_id","speaker_id","target","y_true","pred"]])
        tables[name]=pd.concat(zz,ignore_index=True)
    metrics=[]; deltas=[]; contrasts=[]; ranks=[]; aligns=[]; rng=np.random.default_rng(seed)
    for target in cfg["targets"]:
        ts={k:v[v.target==target].copy() for k,v in tables.items()}
        common=set.intersection(*[set(v.sample_id.astype(str)) for v in ts.values()]); ref=None; aligned={}
        for name,z in ts.items():
            z=z[z.sample_id.astype(str).isin(common)].sort_values("sample_id").copy()
            key=z[["sample_id","speaker_id","y_true"]].reset_index(drop=True)
            if ref is None: ref=key
            else:
                if not np.array_equal(key.sample_id.to_numpy(),ref.sample_id.to_numpy()) or not np.array_equal(key.speaker_id.astype(str).to_numpy(),ref.speaker_id.astype(str).to_numpy()) or not np.allclose(key.y_true,ref.y_true,atol=1e-8,rtol=0): raise RuntimeError(f"alignment fail {name}/{target}")
            sp,mm=moments(z); aligned[name]=(sp,mm)
        speakers=next(iter(aligned.values()))[0]; ns=len(speakers)
        if any(v[0]!=speakers for v in aligned.values()): raise RuntimeError("speaker mismatch")
        counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float); boot={}
        aligns.append({"target":target,"common_samples":len(common),"n_speakers":ns,**{f"n_{k}":len(v) for k,v in ts.items()}})
        for name,(sp,mm) in aligned.items():
            boot[name]={}
            for m in ["overall","between","within"]:
                vals=ccc((counts@mm[m])/ns); boot[name][m]=vals; lo,hi=qci(vals)
                metrics.append({"target":target,"model":name,"metric":m,"ccc":point(mm[m]),"ci95_low":lo,"ci95_high":hi})
        for m in ["overall","between","within"]:
            scores={n:point(v[1][m]) for n,v in aligned.items()}
            ranks.append({"target":target,"metric":m,"ranking":" > ".join(sorted(scores,key=scores.get,reverse=True)),**scores})
            for x,y in itertools.combinations(sorted(aligned),2):
                dv=boot[x][m]-boot[y][m]; lo,hi=qci(dv)
                deltas.append({"target":target,"metric":m,"comparison":f"{x}-{y}","delta_ccc":scores[x]-scores[y],"ci95_low":lo,"ci95_high":hi,"ci_excludes_zero":bool(lo>0 or hi<0)})
        for x,y in itertools.combinations(sorted(aligned),2):
            dv=(boot[x]["between"]-boot[y]["between"])-(boot[x]["within"]-boot[y]["within"]); lo,hi=qci(dv)
            pt=(point(aligned[x][1]["between"])-point(aligned[y][1]["between"]))-(point(aligned[x][1]["within"])-point(aligned[y][1]["within"]))
            contrasts.append({"target":target,"comparison":f"{x}-{y}","contrast":"between_minus_within","delta_of_deltas":pt,"ci95_low":lo,"ci95_high":hi,"ci_excludes_zero":bool(lo>0 or hi<0)})
    pd.DataFrame(aligns).to_csv(root/"panel_sample_alignment.csv",index=False)
    pd.DataFrame(metrics).to_csv(root/"panel_component_metrics.csv",index=False)
    pd.DataFrame(deltas).to_csv(root/"panel_pairwise_component_deltas.csv",index=False)
    pd.DataFrame(contrasts).to_csv(root/"panel_profile_contrasts.csv",index=False)
    pd.DataFrame(ranks).to_csv(root/"panel_rank_summary.csv",index=False)
    (root/"panel_metadata.json").write_text(json.dumps({"bootstrap_unit":"speaker","bootstrap_reps":reps,"bootstrap_seed":seed,"prediction_condition":"absolute","relative_role":"diagnostic_only"},indent=2)+"\n")
    print(pd.DataFrame(ranks).to_string(index=False)); print("\nSIG PROFILE CONTRASTS"); print(pd.DataFrame(contrasts).query("ci_excludes_zero").to_string(index=False))
if __name__=="__main__": main()

