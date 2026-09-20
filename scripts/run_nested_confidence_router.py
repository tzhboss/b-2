#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC=[
 "class_prevalence","speaker_coverage","absolute_effect_signed","relative_effect_signed",
 "baseline_effect_signed","absolute_effect_abs","relative_effect_abs","baseline_effect_abs",
 "relative_minus_absolute_separation","baseline_prior_corr",
 "between_baseline_to_within_relative_sd_ratio","conflict_fraction"
]

def fit_prob(train,test,C):
    model=Pipeline([
      ("scale",StandardScaler()),
      ("clf",LogisticRegression(C=float(C),class_weight="balanced",max_iter=5000,solver="lbfgs"))
    ])
    model.fit(train[NUMERIC],train.target_positive)
    classes=list(model.named_steps["clf"].classes_)
    pos_idx=classes.index(1)
    return model.predict_proba(test[NUMERIC])[:,pos_idx]

def best_fixed(train,cell_map):
    ids=train[["dataset","attribute","emotion"]]
    g=ids.merge(cell_map,on=["dataset","attribute","emotion"],how="left",validate="one_to_one")
    return "relative" if g.relative.mean()>=g.absolute.mean() else "absolute"

def route(prob,threshold,fallback):
    if prob>=threshold: return "relative"
    if prob<=1-threshold: return "absolute"
    return fallback

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg["outputs"]["artifact_root"]); root.mkdir(parents=True,exist_ok=True)
    x=pd.read_csv(cfg["sources"]["effect_matrix"])
    cells=pd.read_csv(cfg["sources"]["router_cells"])[
      ["dataset","attribute","emotion","absolute","relative","oracle_f1","stats_only_router_f1"]
    ].copy()
    if len(x)!=75 or len(cells)!=75:
        raise RuntimeError("expected 75 cells")

    thresholds=[float(t) for t in cfg["parameters"]["thresholds"]]
    C=float(cfg["parameters"]["logistic_C"])
    outer_rows=[]; inner_rows=[]

    corpora=sorted(x.dataset.unique())
    for outer in corpora:
        outer_train=x[x.dataset.ne(outer)].copy()
        outer_test=x[x.dataset.eq(outer)].copy()

        # Inner out-of-corpus predictions for threshold selection.
        candidate_scores=[]
        for t in thresholds:
            achieved=[]
            for inner in sorted(outer_train.dataset.unique()):
                itr=outer_train[outer_train.dataset.ne(inner)].copy()
                ite=outer_train[outer_train.dataset.eq(inner)].copy()
                fallback=best_fixed(itr,cells)
                prob=fit_prob(itr,ite,C)
                ite_keys=ite[["dataset","attribute","emotion"]].copy()
                ite_keys["prob"]=prob
                g=ite_keys.merge(cells,on=["dataset","attribute","emotion"],how="left",validate="one_to_one")
                for _,r in g.iterrows():
                    choice=route(float(r.prob),t,fallback)
                    achieved.append(float(r[choice]))
            candidate_scores.append((t,float(np.mean(achieved))))
            inner_rows.append({
                "outer_held_out":outer,"threshold":t,
                "inner_mean_achieved_f1":float(np.mean(achieved))
            })
        best_score=max(s for _,s in candidate_scores)
        chosen=max(t for t,s in candidate_scores if abs(s-best_score)<1e-12)

        fallback=best_fixed(outer_train,cells)
        prob=fit_prob(outer_train,outer_test,C)
        out=outer_test[["dataset","attribute","emotion"]].copy()
        out["prob_relative"]=prob
        out=out.merge(cells,on=["dataset","attribute","emotion"],how="left",validate="one_to_one")
        for _,r in out.iterrows():
            choice=route(float(r.prob_relative),chosen,fallback)
            achieved=float(r[choice])
            fixed=float(r[fallback])
            outer_rows.append({
              "dataset":r.dataset,"attribute":r.attribute,"emotion":r.emotion,
              "selected_threshold":chosen,"fallback":fallback,
              "prob_relative":float(r.prob_relative),"nested_choice":choice,
              "nested_f1":achieved,"training_best_fixed_f1":fixed,
              "naive_stats_router_f1":float(r.stats_only_router_f1),
              "oracle_f1":float(r.oracle_f1),
              "absolute":float(r.absolute),"relative":float(r.relative)
            })

    out=pd.DataFrame(outer_rows); inner=pd.DataFrame(inner_rows)
    rows=[]
    for policy,col in [
      ("training_best_fixed","training_best_fixed_f1"),
      ("naive_stats_router","naive_stats_router_f1"),
      ("nested_router","nested_f1"),
      ("oracle","oracle_f1")
    ]:
        rows.append({
          "policy":policy,"n_cells":len(out),
          "mean_f1":float(out[col].mean()),
          "mean_oracle_regret":float((out.oracle_f1-out[col]).mean())
        })
    summary=pd.DataFrame(rows)
    best_fixed=float(summary[summary.policy.eq("training_best_fixed")].mean_f1.iloc[0])
    summary["gain_vs_training_best_fixed"]=summary.mean_f1-best_fixed

    by=[]
    for ds,g in out.groupby("dataset"):
        for policy,col in [
          ("training_best_fixed","training_best_fixed_f1"),
          ("naive_stats_router","naive_stats_router_f1"),
          ("nested_router","nested_f1"),
          ("oracle","oracle_f1")
        ]:
            by.append({
              "dataset":ds,"policy":policy,"n_cells":len(g),
              "mean_f1":float(g[col].mean()),
              "mean_oracle_regret":float((g.oracle_f1-g[col]).mean()),
              "gain_vs_training_best_fixed":float(g[col].mean()-g.training_best_fixed_f1.mean())
            })
    by=pd.DataFrame(by)

    out.to_csv(root/"nested_router_cells.csv",index=False)
    inner.to_csv(root/"inner_threshold_selection.csv",index=False)
    summary.to_csv(root/"nested_router_summary.csv",index=False)
    by.to_csv(root/"nested_router_by_corpus.csv",index=False)
    (root/"run_metadata.json").write_text(json.dumps({
      "experiment_id":cfg["experiment_id"],
      "thresholds":thresholds,"logistic_C":C,
      "outer_corpora":corpora,
      "selection":"inner leave-one-training-corpus-out"
    },indent=2)+"\n")
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nBY CORPUS")
    print(by.to_string(index=False))
    print("\nTHRESHOLDS")
    print(out.groupby("dataset")[["selected_threshold","fallback"]].first().to_string())

if __name__=="__main__":
    main()
