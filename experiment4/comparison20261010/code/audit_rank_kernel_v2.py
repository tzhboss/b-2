"""Phase24 experiment4 audit: released RankTrunc kernel vs local/global guard.

V1 local-tail guard, V2 full guard; all share exact model, requests, TRAIN split.
No TEST, no post-hoc retuning. Bootstrap exploratory, clustered by query.
"""
import argparse,json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np

def boot(keys,diffs):
    group=defaultdict(list)
    for k,d in zip(keys,diffs):group[tuple(k)].append(float(d))
    vals=np.array([np.mean(v) for v in group.values()])
    if not len(vals):return None
    rng=np.random.default_rng(2451)
    sample=vals[rng.integers(0,len(vals),size=(5000,len(vals)))].mean(axis=1)
    return {"n_unique_queries":len(group),"query_cluster_bootstrap_95":[float(x) for x in np.quantile(sample,[.025,.975])],
            "query_balanced_effect":float(vals.mean())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--work',type=Path,default=Path('.'))
    args=ap.parse_args();r=args.work
    out=[];paired={}
    for kind in ("DistMult","ComplEx"):
        cases=[]
        for seed in (401,402,403):
            a=json.loads((r/"results"/f"{kind.lower()}_seed{seed}_ep25_edit.json").read_text())
            b=json.loads((r/"results"/f"{kind.lower()}_seed{seed}_ep25_edit_v2.json").read_text())
            assert a["n_requests"]==b["n_requests"]==27
            assert a["checkpoint_sha256"]==b["checkpoint_sha256"]
            for v1,v2 in zip(a["rows"],b["rows"]):
                assert v1["query"]==v2["query"] and v1["targets"]==v2["targets"]
                assert v1["n_guard"]==v2["n_guard"] and v1["n_heldother"]==v2["n_heldother"]
                for k1,k2 in (("0","S0"),):
                    for k in ("newhits","cross_damage","cost_l21","old_damage","same_damage"):
                        assert abs(v1["variants"][k1][k]-v2["variants"][k2][k])<1e-6,(kind,seed,k,v1["query"])
                cases.append((seed,v1,v2))
        assert len(cases)==81
        for B in (3.5,5.2):
            method_names=("Direct","Soft3_LocalGuard","Soft3_AllGuard","Rank125_ReleasedKernel",
                          "Rank1of64_ReleasedKernel","AdaptiveSoft_LocalGuard","AdaptiveSoft_AllGuard",
                          "AdaptiveRank_ReleasedKernel")
            bymethod={}
            for name in method_names:
                selected=[];success=[]
                for seed,v1,v2 in cases:
                    if name=="Direct":m=v2["variants"]["S0"]
                    elif name=="Soft3_LocalGuard":m=v1["variants"]["3"]
                    elif name=="Soft3_AllGuard":m=v2["variants"]["S3"]
                    elif name=="Rank125_ReleasedKernel":m=v2["variants"]["R0.125"]
                    elif name=="Rank1of64_ReleasedKernel":m=v2["variants"]["R0.015625"]
                    elif name=="AdaptiveSoft_LocalGuard":
                        m=v1["adaptive"][str(B)]["selected"]
                    elif name=="AdaptiveSoft_AllGuard":
                        m=v2["adaptive"][str(B)]["S"]["selected"]
                    elif name=="AdaptiveRank_ReleasedKernel":
                        m=v2["adaptive"][str(B)]["R"]["selected"]
                    if m is not None and m["all3"]==1 and m["cost_l21"]<=B+1e-8:
                        selected.append(m);success.append(1)
                    else:
                        selected.append(None);success.append(0)
                n=sum(success)
                events=sum(x["cross_damage"] for x in selected if x is not None)
                norm=sum(x["cost_l21"] for x in selected if x is not None)
                held=sum(v2["n_heldother"] for seed,v1,v2 in cases)
                obj={"kind":kind,"budget":B,"method":name,"attempts":81,
                    "n_success":n,"success_rate":n/81,
                    "cross_damage_events":events,"cross_damage_per_attempt":events/81,
                    "cross_damage_per_success":events/n if n else None,
                    "L21_per_attempt":norm/81,"L21_per_success":norm/n if n else None,
                    "mean_same_damage_per_attempt":sum(x["same_damage"] for x in selected if x is not None)/81,
                    "max_accepted_update_cost":max((x["cost_l21"] for x in selected if x is not None),default=0),
                    "heldout_fact_exposures":held,"lost_per_fact_exposure":events/held if held else None,
                    "n_zero_cross_holdout":sum(v2["n_heldother"]==0 for seed,v1,v2 in cases)}
                assert obj["max_accepted_update_cost"]<=B+1e-8
                out.append(obj);bymethod[name]=selected
            def harm(x):return x["cross_damage"] if x else 0
            diffs={}
            for name in ("Soft3_LocalGuard","Soft3_AllGuard","Rank125_ReleasedKernel",
                         "AdaptiveSoft_LocalGuard","AdaptiveSoft_AllGuard","AdaptiveRank_ReleasedKernel"):
                a=bymethod[name];b=bymethod["Direct"]
                d=[harm(x)-harm(y) for x,y in zip(a,b)]
                diffs[name]={"mean_delta_vs_direct":float(np.mean(d)),
                             "improved":sum(k<0 for k in d),"worsened":sum(k>0 for k in d),
                             "ties":sum(k==0 for k in d),
                             "matched_success":sum((x is not None)==(y is not None) for x,y in zip(a,b)),
                             "exploratory":boot([v2["query"] for seed,v1,v2 in cases],d)}
            paired[f"{kind}_B{B}"]=diffs
    dst=r/"results"/"phase24_v2_rank_kernel_audit.json"
    dst.write_text(json.dumps({"source":"author released Rank-Truncated numerical kernel adapted to local shared multi-target protocol",
         "not_full_SOTA_reproduction":True,"TEST_used":False,"n_model_seeds":3,
         "n_eval_per_seed":27,"n_models":6,
         "old_fact_scope":"originally correct TRAIN factual queries heldout, not official TEST",
         "model_dependence":"guard and heldout fact subsets selected per model; do not compare raw cross damage across different model families as if fact cohorts identical",
         "warning":"hard budget comparisons include abstentions with zero damage; always report success together",
         "rows":out,"paired":paired},ensure_ascii=False,indent=2),encoding="utf-8")
    with (r/"results"/"phase24_v2_rank_kernel_audit.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print("V2_AUDIT_OK models=6 cases=162 summaries",len(out))
    for x in out:
        if x["budget"]==5.2:
            print(x["kind"],x["method"],"success",x["n_success"],"cross",round(x["cross_damage_per_attempt"],3),
                  "l21",round(x["L21_per_attempt"],3))
if __name__=="__main__":main()
