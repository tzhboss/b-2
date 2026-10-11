"""Phase28 REAL KGE edit replay: rank-margin based *TRAIN guard* selector.

One fixed 25-epoch checkpoint per seed/family; same heldout EVAL cases as phase24.
Shared entity embeddings act in both heads/tails.
NO official TEST, no fitting/tuning on TRAIN-heldout rank labels.

All methods select from EXACT SAME seven local-soft candidate updates:
lambda in (0,0.3,1,2,3,5,10), and must preserve 3/3 VALID target
top10 and actual per-case L21 <= min(5.2, direct L21+extra).
If Direct infeasible ALL methods atomically abstain.

Risk(s,t) = softplus((10th highest OTHER tail score - true tail score) /
                       max(std(top20 OTHER pre-edit scores), .2))
scores after EDIT are exact for TRAIN GUARD queries; expensive full
matrix multiplication only if modified entity is query head.
This is a hand-designed rank-aware proxy, not a published SOTA reproduction.
"""
import argparse,time,hashlib,json,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
for module_dir in (Path(__file__).resolve().parent.parent.parent/"code",
                   Path(__file__).resolve().parent.parent/"phase24"):
    if (module_dir/"eval_paired_kge_edit.py").exists():
        sys.path.insert(0,str(module_dir));break
import eval_paired_kge_edit as b

LAMBDAS=(0.,.3,1.,2.,3.,5.,10.)
EXTRAS=(.25,.5)
BUDGET=5.2
METHODS=("Direct","FixedSoft3","MinGuardDrift","MarginCross","MarginAll","RankFailCross")

def margin(score,tail):
    # Explicitly exclude target itself when computing 10th-ranked competing tail.
    other=np.concatenate([score[:tail],score[tail+1:]])
    threshold=float(np.partition(other,-10)[-10])
    return float(score[tail]-threshold)

def confidence_scale(base,tail):
    arr=np.concatenate([base[:tail],base[tail+1:]])
    return max(float(np.std(np.partition(arr,-20)[-20:])),.2)

def risk_metrics(E,R,kind,edited,guards,reference):
    dif=np.where(np.any(edited!=E,axis=1))[0]
    changes=set(map(int,dif))
    summaries={"cross":[],"all":[]}
    for (h,rel,t),base,scale,was_cross in reference:
        h=int(h);rel=int(rel);t=int(t)
        if h in changes:
            post=b.scores(edited,R,h,rel,kind)
        else:
            post=base.copy()
            q=b.qvec(E,R,h,rel,kind)
            for i in dif:
                post[int(i)]=float(q@edited[int(i)])
        marg=margin(post,t)
        softplus=float(np.logaddexp(0.,-marg/scale))
        summaries["all"].append((softplus,int(marg<=0)))
        if was_cross:summaries["cross"].append((softplus,int(marg<=0)))
    return {
      "margin_cross":float(np.mean([v[0] for v in summaries["cross"]])) if summaries["cross"] else 0.,
      "margin_all":float(np.mean([v[0] for v in summaries["all"]])) if summaries["all"] else 0.,
      "rankfail_cross":sum(v[1] for v in summaries["cross"]),
      "rankfail_all":sum(v[1] for v in summaries["all"]),
      "n_guard_cross":len(summaries["cross"]),
      "n_guard_all":len(summaries["all"])}

def eval_one(kind,seed,epoch,work,limit):
    start=time.monotonic()
    ck=work/"checkpoints"/f"{kind.lower()}_seed{seed}_epoch{epoch}.pt"
    chk=torch.load(ck,map_location="cpu",weights_only=True)
    assert chk["kind"]==kind and chk["seed"]==seed and chk["epoch"]==epoch
    E,R=b.representations(chk)
    tr,va=b.read_data(work/"data")
    known=defaultdict(set);vgroup=defaultdict(set)
    bytail=defaultdict(list);reltrain=defaultdict(list)
    for hh,rr,t in tr:
        hh,rr,t=map(int,(hh,rr,t))
        known[(hh,rr)].add(t);bytail[t].append((hh,rr,t));reltrain[rr].append((hh,t))
    for hh,rr,t in va:vgroup[(int(hh),int(rr))].add(int(t))
    cases=json.loads((work/"data"/f"phase22_seed{seed}.json").read_text(encoding="utf-8"))["rows"][12:39]
    p24=json.loads((work/"results"/f"{kind.lower()}_seed{seed}_ep{epoch}_edit.json").read_text(encoding="utf-8"))
    assert len(cases)==len(p24["rows"])==27
    assert p24["checkpoint_sha256"]==hashlib.sha256(ck.read_bytes()).hexdigest()
    outputs=[]
    for ci,(ref,previous) in enumerate(zip(cases,p24["rows"])):
        if ci>=limit:break
        h,rel=map(int,ref["query"])
        targets=sorted(vgroup[(h,rel)]-known[(h,rel)])[:3]
        assert len(targets)==3 and targets==previous["targets"]
        case=(h,rel,targets,sorted(known[(h,rel)]))
        guard,heldother,heldsame=b.split_guard(tr,case,E,R,kind,bytail,reltrain)
        assert len(guard)==previous["n_guard"] and len(heldother)==previous["n_heldother"]
        # Retain only legitimate TRAIN guards. TRAIN heldout is not inspected here.
        assert not set(guard)&set(heldother+heldsame)
        reference=[]
        for gh,gr,gt in guard:
            score=b.scores(E,R,int(gh),int(gr),kind)
            assert b.top10(score,int(gt))
            reference.append(((int(gh),int(gr),int(gt)),score,
                              confidence_scale(score,int(gt)),int(gr)!=rel))
        candidates={}
        for lam in LAMBDAS:
            ep=b.edit(E,R,kind,case,guard,lam)
            risk=risk_metrics(E,R,kind,ep,guard,reference)
            # PREVIOUS metrics are from actual identical edits (check direct and
            # fixed 3); selection may read only success, cost, guard_drift, risk.
            saved=previous["variants"][f"{lam:g}"]
            if lam in (0.,3.):
                check=b.metrics(E,R,kind,ep,case,guard,heldother,heldsame)
                for field in ("all3","cost_l21","cross_damage","same_damage","old_damage","guard_drift"):
                    assert abs(saved[field]-check[field])<1e-7,(kind,seed,ci,lam,field)
            candidates[f"{lam:g}"]={"risk":risk,"metrics":saved}
        direct=candidates["0"]["metrics"]
        rows={}
        for extra in EXTRAS:
            cap=min(BUDGET,direct["cost_l21"]+extra)
            possible=[(k,c) for k,c in candidates.items()
                     if direct["all3"] and direct["cost_l21"]<=BUDGET+1e-8
                     and c["metrics"]["all3"] and c["metrics"]["cost_l21"]<=cap+1e-8]
            chosen={}
            for method in METHODS:
                if method=="Direct":
                    item=next((x for x in possible if x[0]=="0"),None)
                elif method=="FixedSoft3":
                    item=next((x for x in possible if x[0]=="3"),None)
                    if item is None:item=next((x for x in possible if x[0]=="0"),None)
                else:
                    def key(x):
                        k,rec=x
                        if method=="MinGuardDrift":first=rec["metrics"]["guard_drift"]
                        elif method=="MarginCross":first=rec["risk"]["margin_cross"]
                        elif method=="MarginAll":first=rec["risk"]["margin_all"]
                        elif method=="RankFailCross":first=rec["risk"]["rankfail_cross"]
                        else:raise KeyError(method)
                        return (first,rec["metrics"]["cost_l21"],float(k))
                    item=min(possible,key=key) if possible else None
                m=item[1]["metrics"] if item else None
                chosen[method]={
                    "lambda":float(item[0]) if item else None,
                    "all3":m["all3"] if m else 0,
                    "cross_damage":m["cross_damage"] if m else 0,
                    "same_damage":m["same_damage"] if m else 0,
                    "old_damage":m["old_damage"] if m else 0,
                    "cost_l21":m["cost_l21"] if m else 0.,
                    "guard_drift":m["guard_drift"] if m else None,
                    "rank_risk":item[1]["risk"] if item else None}
                if m:
                    assert m["all3"] and m["cost_l21"]<=cap+1e-8
            assert len({v["all3"] for v in chosen.values()})==1
            rows[str(extra)]={"direct_cap":cap,"methods":chosen}
        outputs.append({"seed":seed,"case_index":ci,"query":[h,rel],"targets":targets,
                         "n_crossheld":len(heldother),"n_sameheld":len(heldsame),
                         "n_guard":len(guard),"candidates":candidates,
                         "results":rows})
        if (ci+1)%5==0:
            print("PROGRESS",kind,seed,epoch,ci+1,"elapsed_s",round(time.monotonic()-start,1),flush=True)
    dest=work/"phase28"/"results"/f"{kind.lower()}_seed{seed}_ep{epoch}_rankrisk.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    result={"kind":kind,"seed":seed,"epoch":epoch,
      "checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),
      "n_eval":len(outputs),"TRAIN_guard_only_selection":True,"TEST_used":False,
      "no_train_heldout_in_risk_proxy":True,
      "selection_rule_locked":"rank risk softplus against exact 10th OTHER competitor / max(std(top20 OTHER original scores),.2)",
      "methods":METHODS,"lambdas":LAMBDAS,"extras":EXTRAS,"budget":BUDGET,
      "source":"exact phase24 candidate edits replayed on remote trained checkpoint; post-hoc candidate metric equality verified",
      "not_full_author_sota":True,"elapsed_s":time.monotonic()-start,"rows":outputs}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("SAVED",dest,"ROWS",len(outputs),"ELAPSED",round(result["elapsed_s"],1),flush=True)
    for extra in EXTRAS:
        for method in METHODS:
            values=[x["results"][str(extra)]["methods"][method] for x in outputs]
            print("RESULT",kind,seed,extra,method,"succ",sum(x["all3"] for x in values),
                  "cross",sum(x["cross_damage"] for x in values),
                  "meanCost",round(float(np.mean([x["cost_l21"] for x in values])),4),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--work",type=Path,default=Path("."))
    ap.add_argument("--kind",choices=("DistMult","ComplEx"),required=True)
    ap.add_argument("--seed",type=int,choices=(401,402,403),required=True)
    ap.add_argument("--epoch",type=int,default=25)
    ap.add_argument("--limit",type=int,default=27)
    a=ap.parse_args()
    eval_one(a.kind,a.seed,a.epoch,a.work,a.limit)
