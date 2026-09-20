#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

POLICIES=[
    "always_absolute","always_relative","labels_only_router",
    "stats_only_router","combined_router","oracle_router"
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)

    pred=pd.read_csv(cfg["sources"]["loco_predictions"])
    pred=pred[pred.split_type.eq("dataset")].copy()

    p9=pd.read_csv(cfg["sources"]["pitch_summary"])
    p9=p9[p9.representation.isin(["absolute","relative"])].copy()
    p9["attribute"]="pitch"
    p10=pd.read_csv(cfg["sources"]["multi_attribute_summary"])
    p10=p10[p10.representation.isin(["absolute","relative"])].copy()

    summaries=pd.concat([
        p9[["dataset","attribute","emotion","representation","speaker_balanced_f1_mean"]],
        p10[["dataset","attribute","emotion","representation","speaker_balanced_f1_mean"]]
    ],ignore_index=True)
    wide=summaries.pivot_table(
        index=["dataset","attribute","emotion"],
        columns="representation",values="speaker_balanced_f1_mean"
    ).reset_index()
    wide.columns.name=None
    if len(wide)!=75:
        raise RuntimeError(f"expected 75 F1 cells, got {len(wide)}")
    wide["oracle_f1"]=wide[["absolute","relative"]].max(axis=1)
    wide["oracle_choice_relative"]=(wide.relative>wide.absolute).astype(int)

    resolved=pred[pred.model_family.eq("stats_only")][
        ["dataset","attribute","emotion","resolved"]
    ].drop_duplicates()
    cells=wide.merge(resolved,on=["dataset","attribute","emotion"],how="left",validate="one_to_one")

    for fam in ["labels_only","stats_only","combined"]:
        q=pred[pred.model_family.eq(fam)][
            ["dataset","attribute","emotion","predicted_positive"]
        ].rename(columns={"predicted_positive":f"{fam}_choose_relative"})
        cells=cells.merge(q,on=["dataset","attribute","emotion"],how="left",validate="one_to_one")

    cells["always_absolute_f1"]=cells.absolute
    cells["always_relative_f1"]=cells.relative
    cells["labels_only_router_f1"]=np.where(cells.labels_only_choose_relative.eq(1),cells.relative,cells.absolute)
    cells["stats_only_router_f1"]=np.where(cells.stats_only_choose_relative.eq(1),cells.relative,cells.absolute)
    cells["combined_router_f1"]=np.where(cells.combined_choose_relative.eq(1),cells.relative,cells.absolute)
    cells["oracle_router_f1"]=cells.oracle_f1

    rows=[]
    for subset_name,mask in [("all",np.ones(len(cells),bool)),("resolved",cells.resolved.fillna(False).to_numpy(bool))]:
        sub=cells.loc[mask]
        fixed_best=max(sub.absolute.mean(),sub.relative.mean())
        for policy in POLICIES:
            col=f"{policy}_f1"
            mean=float(sub[col].mean())
            regret=float((sub.oracle_f1-sub[col]).mean())
            choice_match=np.nan
            if policy=="always_absolute":
                choice=(np.zeros(len(sub),int))
            elif policy=="always_relative":
                choice=np.ones(len(sub),int)
            elif policy=="oracle_router":
                choice=sub.oracle_choice_relative.to_numpy(int)
            else:
                fam=policy.replace("_router","")
                choice=sub[f"{fam}_choose_relative"].to_numpy(int)
            choice_match=float(np.mean(choice==sub.oracle_choice_relative.to_numpy(int)))
            rows.append({
                "subset":subset_name,"policy":policy,"n_cells":len(sub),
                "mean_achieved_f1":mean,"mean_oracle_regret":regret,
                "fraction_oracle_choice":choice_match,
                "gain_vs_always_absolute":mean-float(sub.absolute.mean()),
                "gain_vs_always_relative":mean-float(sub.relative.mean()),
                "gain_vs_better_fixed":mean-fixed_best
            })
    summary=pd.DataFrame(rows)

    by=[]
    for ds,g in cells.groupby("dataset"):
        fixed_best=max(g.absolute.mean(),g.relative.mean())
        for policy in POLICIES:
            mean=float(g[f"{policy}_f1"].mean())
            by.append({
                "dataset":ds,"policy":policy,"n_cells":len(g),
                "mean_achieved_f1":mean,
                "mean_oracle_regret":float((g.oracle_f1-g[f"{policy}_f1"]).mean()),
                "gain_vs_better_fixed":mean-fixed_best
            })
    by=pd.DataFrame(by)

    cells.to_csv(root/"router_cells.csv",index=False)
    summary.to_csv(root/"router_summary.csv",index=False)
    by.to_csv(root/"router_by_corpus.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
        "experiment_id":cfg["experiment_id"],
        "n_cells":len(cells),
        "n_resolved":int(cells.resolved.sum()),
        "router_source":"EXP-11 leave-one-corpus-out predictions",
        "f1_sources":["EXP-09 pitch","EXP-10 loudness/rate"]
    },indent=2)+"\n")

    print("ROUTER SUMMARY")
    print(summary.to_string(index=False))
    print("\nBY CORPUS")
    print(by.to_string(index=False))

if __name__=="__main__":
    main()
