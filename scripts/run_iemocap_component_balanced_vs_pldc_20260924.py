#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, itertools, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_iemocap_paired_crossmodel_component_attribution_20260924 import ccc, moments, point, qci

METHOD_SOURCES = {
    "nested_best_single": "exp11",
    "inner_convex_ssl_ensemble": "exp11",
    "pldc_mu": "exp10",
    "pldc_sigma": "exp10",
    "pldc_mu_sigma": "exp10",
    "component_balanced_maximin": "exp11",
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    root=Path(cfg["output_root"])
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True, exist_ok=True)

    d10=pd.read_parquet(cfg["exp10_oof"])
    d11=pd.read_parquet(cfg["exp11_oof"])
    sources={"exp10":d10, "exp11":d11}
    reps=int(cfg["bootstrap_reps"]); seed=int(cfg["bootstrap_seed"])
    rng=np.random.default_rng(seed)
    metric_rows=[]; delta_rows=[]; summary_rows=[]; align_rows=[]

    for target in ["arousal","dominance"]:
        tables={}; ref=None
        for method,src in METHOD_SOURCES.items():
            z=sources[src]
            z=z[(z.target==target)&(z.method==method)][["sample_id","speaker_id","y_true","pred"]].copy()
            z=z.sort_values("sample_id").reset_index(drop=True)
            if ref is None:
                ref=z[["sample_id","speaker_id","y_true"]].copy()
            else:
                if not np.array_equal(z.sample_id.astype(str),ref.sample_id.astype(str)):
                    raise RuntimeError(f"sample alignment failed {target}/{method}")
                if not np.array_equal(z.speaker_id.astype(str),ref.speaker_id.astype(str)):
                    raise RuntimeError(f"speaker alignment failed {target}/{method}")
                if not np.allclose(z.y_true.to_numpy(float),ref.y_true.to_numpy(float),atol=1e-10,rtol=0):
                    raise RuntimeError(f"truth alignment failed {target}/{method}")
            tables[method]=z
        align_rows.append({"target":target,"samples":len(ref),"speakers":int(ref.speaker_id.nunique()),
                           **{f"n_{m}":len(z) for m,z in tables.items()}})

        aligned={}; boot={}; speakers=None
        for method,z in tables.items():
            sp,mm=moments(z)
            if speakers is None: speakers=sp
            elif sp!=speakers: raise RuntimeError("speaker order mismatch")
            aligned[method]=(sp,mm)
        ns=len(speakers)
        counts=rng.multinomial(ns,np.full(ns,1/ns),size=reps).astype(float)

        for method,(_,mm) in aligned.items():
            boot[method]={}
            for metric in ["overall","between","within"]:
                vals=ccc((counts@mm[metric])/ns)
                boot[method][metric]=vals
                lo,hi=qci(vals)
                metric_rows.append({"target":target,"method":method,"metric":metric,
                                    "ccc":point(mm[metric]),"ci95_low":lo,"ci95_high":hi})
        for metric in ["overall","between","within"]:
            scores={m:point(mm[metric]) for m,(_,mm) in aligned.items()}
            for x,y in itertools.combinations(METHOD_SOURCES,2):
                dv=boot[x][metric]-boot[y][metric]
                lo,hi=qci(dv)
                delta_rows.append({"target":target,"metric":metric,"comparison":f"{x}-{y}",
                                   "delta_ccc":scores[x]-scores[y],"ci95_low":lo,"ci95_high":hi,
                                   "ci_excludes_zero":bool(lo>0 or hi<0)})

        maximin="component_balanced_maximin"
        for refm in ["nested_best_single","inner_convex_ssl_ensemble","pldc_mu_sigma"]:
            for metric in ["overall","between","within"]:
                rec=[r for r in delta_rows if r["target"]==target and r["metric"]==metric
                     and r["comparison"]==f"{maximin}-{refm}"][0]
                summary_rows.append({"target":target,"method":maximin,"reference":refm,
                                     **{k:rec[k] for k in ["metric","delta_ccc","ci95_low","ci95_high","ci_excludes_zero"]}})

    pd.DataFrame(align_rows).to_csv(root/"sample_alignment.csv",index=False)
    pd.DataFrame(metric_rows).to_csv(root/"component_metrics.csv",index=False)
    pd.DataFrame(delta_rows).to_csv(root/"paired_method_deltas.csv",index=False)
    pd.DataFrame(summary_rows).to_csv(root/"maximin_vs_key_controls.csv",index=False)

    sm=pd.DataFrame(summary_rows); checks={}
    for target in ["arousal","dominance"]:
        g=sm[(sm.target==target)&(sm.reference=="nested_best_single")]
        checks[target+"_positive_all_three_vs_best"]=bool((g.delta_ccc>0).all() and len(g)==3)
        p=sm[(sm.target==target)&(sm.reference=="pldc_mu_sigma")&(sm.metric=="overall")]
        e=sm[(sm.target==target)&(sm.reference=="inner_convex_ssl_ensemble")&(sm.metric=="overall")]
        checks[target+"_overall_noninferior_point_to_pldc"]=bool(len(p)==1 and p.delta_ccc.iloc[0]>=0)
        checks[target+"_overall_noninferior_point_to_convex"]=bool(len(e)==1 and e.delta_ccc.iloc[0]>=0)

    decision="pass" if all(checks[t+"_positive_all_three_vs_best"] for t in ["arousal","dominance"]) and all(
        checks[t+"_overall_noninferior_point_to_pldc"] or checks[t+"_overall_noninferior_point_to_convex"]
        for t in ["arousal","dominance"]) else "mixed"

    audit={
        "experiment_id":cfg["experiment_id"],"validity":"valid","decision":decision,
        "task":"raw absolute IEMOCAP Arousal/Dominance","split":"identical nested 5-session LOSO",
        "bootstrap_unit":"speaker","bootstrap_reps":reps,"methods":list(METHOD_SOURCES),
        "checks":checks,
        "interpretation":"Direct paired common-OOF comparison against published PLDC mu+sigma and strong generic convex ensemble; no retraining or post-result tuning in this experiment."
    }
    (root/"audit_summary.json").write_text(json.dumps(audit,indent=2)+"\n")
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "sources":{"exp10":cfg["exp10_oof"],"exp11":cfg["exp11_oof"]},
        "bootstrap_unit":"speaker","bootstrap_reps":reps,"bootstrap_seed":seed
    },indent=2)+"\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
