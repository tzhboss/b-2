"""Experiment4 aggregate and audit: exact fixed phase22 query keys, hard L2,1,
shared head/tail, no TEST; independent TRAIN split and validation caveats."""
import argparse,json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
KINDS=("DistMult","ComplEx");SEEDS=(401,402,403);BUDGETS=(3.5,5.2)
def audit(root,epoch):
    raw={}
    for kind in KINDS:
        for seed in SEEDS:
            f=root/"results"/f"{kind.lower()}_seed{seed}_ep{epoch}_edit.json"
            x=json.loads(f.read_text(encoding='utf-8'))
            prior=json.loads((root/"data"/f"phase22_seed{seed}.json").read_text(encoding='utf-8'))
            assert x["seed"]==seed and x["kind"]==kind and x["n_requests"]==27
            assert not x["TEST_used"] and x["model_shared_head_tail"]
            assert [r["query"] for r in x["rows"]]==[r["query"] for r in prior["rows"][12:39]]
            assert all(len(r["targets"])==3 for r in x["rows"])
            for row in x["rows"]:
                for label,metric in row["variants"].items():
                    assert metric["all3"]==int(metric["newhits"]==3)
                    assert metric["cost_l21"]>=0
                    assert metric["n_heldother"] if "n_heldother" in metric else True
                for B in BUDGETS:
                    z=row["adaptive"][str(B)]
                    if z["accepted"]:
                        assert z["selected"]["all3"]==1 and z["selected"]["cost_l21"]<=B+1e-8
                    else:
                        assert z["selected"] is None
            raw[(kind,seed)]=x
    assert all([raw[("DistMult",s)]["rows"][i]["query"]==
               raw[("ComplEx",s)]["rows"][i]["query"]
               for s in SEEDS for i in range(27)])
    allout=[];paired={}
    for kind in KINDS:
        rows=[r for seed in SEEDS for r in raw[(kind,seed)]["rows"]]
        for B in BUDGETS:
            results={}
            for method in ("Direct","Soft3","AdaptiveSoft"):
                agg=[]
                for z in rows:
                    if method=="AdaptiveSoft":
                        a=z["adaptive"][str(B)]
                        agg.append(a["selected"] if a["accepted"] else None)
                    else:
                        m=z["variants"]["0" if method=="Direct" else "3"]
                        agg.append(m if (m["all3"] and m["cost_l21"]<=B+1e-8) else None)
                hit=sum(m is not None for m in agg)
                total_damage=sum(m["cross_damage"] for m in agg if m is not None)
                total_norm=sum(m["cost_l21"] for m in agg if m is not None)
                total_same=sum(m["same_damage"] for m in agg if m is not None)
                nheld=sum(r["n_heldother"] for r in rows)
                obj={"model":kind,"epoch":epoch,"budget":B,"method":method,"attempts":len(rows),
                     "successful_edits":hit,"success_rate":hit/len(rows),
                     "cross_damage_events":total_damage,"cross_damage_per_attempt":total_damage/len(rows),
                     "same_damage_per_attempt":total_same/len(rows),
                     "cross_heldout_fact_exposures":nheld,
                     "cross_damage_per_heldout_fact_exposure":total_damage/nheld if nheld else None,
                     "L21_per_attempt":total_norm/len(rows),
                     "L21_per_success":total_norm/hit if hit else None,
                     "max_actual_executed_L21":max([m["cost_l21"] if m else 0 for m in agg]),
                     "mean_other_fact_count_per_request":nheld/len(rows),
                     "new_targets_originally_hit_mean":float(np.mean([z["n_original_valid_hits"] for z in rows])),
                     "n_zero_other_heldout":sum(z["n_heldother"]==0 for z in rows),
                     "avg_guard_size":float(np.mean([z["n_guard"] for z in rows]))}
                assert obj["max_actual_executed_L21"]<=B+1e-8
                allout.append(obj)
                results[method]=obj
            # Compare identical evaluation queries, but fact sets are model-dependent.
            paired[f"{kind}_{B}"]={
                "n_queries":len({tuple(z["query"]) for z in rows}),
                "note":"Within model/query hard budget; heldout fact selection model-dependent. Across-model raw damage not directly comparable.",
                "same_accepted_count_as_direct":results["AdaptiveSoft"]["successful_edits"]==results["Direct"]["successful_edits"]}
    target=root/"results"/f"phase24_audit_epoch{epoch}.json"
    target.write_text(json.dumps({"epoch":epoch,"n_model_seeds":3,
      "model_families":list(KINDS),"TEST_used":False,
      "method_authorship":"local methods, not official author SOTA benchmark",
      "warning":"VALID was used as periodic training diagnostic and as edit target; model recipe/hyperparams fixed ahead of edit evaluation; validation not independent model selection. Heldout TRAIN facts are not test set.",
      "rows":allout,"comparability":paired},ensure_ascii=False,indent=2),encoding="utf-8")
    with (root/"results"/f"phase24_audit_epoch{epoch}.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allout[0]));w.writeheader();w.writerows(allout)
    print("AUDIT_OK",epoch,"n_results",len(allout))
    for x in allout:
        if x["budget"]==5.2:
            print(x["model"],x["method"],"success",x["successful_edits"],
                  "cross",round(x["cross_damage_per_attempt"],3),
                  "norm",round(x["L21_per_attempt"],3),
                  "heldfacts",x["cross_heldout_fact_exposures"],
                  "base_newhit",round(x["new_targets_originally_hit_mean"],3))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--work",type=Path,default=Path("."))
    ap.add_argument("--epoch",type=int,default=25)
    args=ap.parse_args();audit(args.work,args.epoch)
