"""Experiment4: paired 3-target KGE edits on DistMult and ComplEx checkpoints.

Fixed case keys from phase22 (27 / training seed, partition EVAL).
Same TRAIN/VALID IDs; no TEST. Rebuild model-dependent original Top10
credible TRAIN guard/holdout; use guard only to choose adaptive lambda.
Real shared head/tail scoring for all methods; no frozen-head shortcut.

This is OUR baseline implementation; NOT full original rank locality paper
and NOT the KGEditor author system.
"""
import argparse,json,hashlib,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"vendor"))
from kge_rank_locality import ProtectionSubspace, rank_truncated_delta
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

LAMBDAS=(0.,.3,1.,2.,3.,5.,10.)
FRACTIONS=(.015625,.03125,.0625,.125,.25)
BUDGETS=(3.5,5.2)

def read_data(root):
    def tab(name):
        return {v:int(k) for k,v in (line.split("\t",1) for line in (root/name).read_text(encoding='utf-8').splitlines())}
    E=tab("entities.dict");R=tab("relations.dict")
    def triples(n):
        return np.asarray([[E[h],R[r],E[t]] for h,r,t in
                           (line.split("\t") for line in (root/n).read_text(encoding='utf-8').splitlines())],dtype=np.int32)
    return triples("train.txt"),triples("valid.txt")

def representations(ck):
    real=ck["entities"].cpu().numpy().astype(np.float64)
    rel=ck["relations"].cpu().numpy().astype(np.float64)
    if ck["kind"]=="ComplEx":
        imag=ck["entities_im"].cpu().numpy().astype(np.float64)
        reli=ck["relations_im"].cpu().numpy().astype(np.float64)
        return np.concatenate([real,imag],axis=1),np.concatenate([rel,reli],axis=1)
    return real,rel

def qvec(e,r,h,rel,kind):
    if kind=="DistMult":return e[h]*r[rel]
    d=e.shape[1]//2
    hr,hi=e[h,:d],e[h,d:];rr,ri=r[rel,:d],r[rel,d:]
    return np.concatenate([hr*rr-hi*ri,hr*ri+hi*rr])

def scores(e,r,h,rel,kind):
    return e@qvec(e,r,h,rel,kind)

def top10(s,t):
    return 1+np.count_nonzero(s>s[int(t)])<=10

def split_guard(train,case,e,r,kind,bytail,reltrain,maxguard=24):
    h,rel,targets,known=case
    # Phase19-inspired deterministic TRAIN-only protection and disjoint
    # heldout. Include only facts the model ranked raw Top10 BEFORE editing.
    others=[];seen=set()
    for t in list(targets)+list(known):
        local=0
        for hh,rr,tt in bytail[int(t)]:
            if rr==rel or (hh,rr,tt) in seen:continue
            if top10(scores(e,r,hh,rr,kind),tt):
                seen.add((hh,rr,tt));others.append((hh,rr,tt));local+=1
                if local>=8:break
    crossguard=others[0::2][:maxguard];crossheld=others[1::2][:maxguard]
    sam=[];headseen={int(h)}
    for hh,tt in reltrain[int(rel)]:
        if hh in headseen:continue
        if top10(scores(e,r,hh,int(rel),kind),tt):
            headseen.add(hh);sam.append((hh,int(rel),tt))
        if len(sam)>=40:break
    sameguard=sam[0::2][:10];sameheld=sam[1::2][:10]
    assert not set(crossguard+sameguard)&set(crossheld+sameheld)
    return crossguard+sameguard,crossheld,sameheld

def guard_for_tail(e,r,kind,guard,tail):
    # same q protection direction as phase19 soft formulation: tail slot
    # Jacobian only; final metrics use fully shared entity embeddings.
    rows=[qvec(e,r,int(h),int(rr),kind) for h,rr,tt in guard if tt==tail]
    if not rows:return np.empty((0,e.shape[1]))
    arr=np.asarray(rows)
    return arr/np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-12)

def edit(e,r,kind,case,guard,lam,mode="soft"):
    h,rel,targets,known=case
    first=scores(e,r,h,rel,kind)
    non=np.asarray([i for i in range(len(e)) if i not in set(targets)],dtype=np.int32)
    positives=[int(t) for t in known if t not in targets and top10(first,t)]
    old=sorted(positives,key=lambda t:first[t],reverse=True)
    best=[t for t in old if np.count_nonzero(first[non]>first[t])<7]
    chosen=best+[x for x in old if x not in best][:max(0,7-len(best))]
    edited=e.copy()
    q=qvec(e,r,h,rel,kind)
    qs=float(q@q)
    if qs<1e-10:return edited
    cache={}
    guard_all=np.asarray([qvec(e,r,int(hh),int(rr),kind) for hh,rr,tt in guard],
                         dtype=np.float64).reshape((-1,len(q)))
    g=guard_all/np.maximum(np.linalg.norm(guard_all,axis=1,keepdims=True),1e-12)
    space=ProtectionSubspace.from_directions(guard_all,dimension=len(q)) if mode=="rank" else None
    def update(t,threshold):
        gap=float(threshold-q@edited[t])
        if gap<=1e-10:return
        if mode=="rank":
            delta,requested,used=rank_truncated_delta(q,gap,space,float(lam))
            edited[t]+=delta
            return
        if lam:
            if t not in cache:
                if len(g):
                    A=np.eye(len(q))+lam*g.T@g
                    vec=np.linalg.solve(A,q)
                    den=float(q@vec)
                    direction=vec/den if den>1e-10 else q/qs
                else:direction=q/qs
                cache[t]=direction
            direction=cache[t]
        else:direction=q/qs
        edited[t]+=gap*direction
    if chosen:
        # preserve selected originally credible old answers first
        disallowed=np.setdiff1d(non,np.asarray(chosen),assume_unique=True)
        spare=7-len(chosen)
        thresh=float(np.partition(first[disallowed],-(spare+1))[-(spare+1)])+1e-5
        for t in chosen:update(t,thresh)
    current=scores(edited,r,h,rel,kind)
    threshold=float(np.partition(current[non],-8)[-8])+1e-5
    for t in targets:update(t,threshold)
    return edited

def metrics(e,r,kind,edited,case,guard,heldother,heldsame):
    h,rel,targets,known=case
    before=scores(e,r,h,rel,kind)
    after=scores(edited,r,h,rel,kind)
    old=[t for t in known if t not in targets and top10(before,t)]
    def fallen(triples):
        return sum(int(not top10(scores(edited,r,int(hh),int(rr),kind),int(tt)))
                   for hh,rr,tt in triples)
    def drift(triples):
        if not triples:return 0.
        pairs=[]
        for hh,rr,tt in triples:
            s0=float(qvec(e,r,int(hh),int(rr),kind)@e[int(tt)])
            s1=float(qvec(edited,r,int(hh),int(rr),kind)@edited[int(tt)])
            pairs.append(((s1-s0)/(1+abs(s0)))**2)
        return float(np.mean(pairs))
    return {"all3":int(sum(top10(after,t) for t in targets)==3),
      "newhits":int(sum(top10(after,t) for t in targets)),
      "original_new_hits":int(sum(top10(before,t) for t in targets)),
      "old_damage":int(sum(not top10(after,t) for t in old)),
      "cross_damage":int(fallen(heldother)),
      "same_damage":int(fallen(heldsame)),
      "guard_drift":drift(guard),
      "cost_l21":float(np.linalg.norm(edited-e,axis=1).sum()),
      "fro":float(np.linalg.norm(edited-e))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--work",type=Path,default=Path("."))
    ap.add_argument("--kind",choices=("DistMult","ComplEx"),required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--epoch",type=int,default=25)
    ap.add_argument("--limit",type=int,default=27)
    args=ap.parse_args()
    np.seterr(all='raise')
    root=args.work;ckpath=root/"checkpoints"/f"{args.kind.lower()}_seed{args.seed}_epoch{args.epoch}.pt"
    ck=torch.load(ckpath,map_location="cpu",weights_only=True)
    assert ck["seed"]==args.seed and ck["kind"]==args.kind
    E,R=representations(ck)
    train,valid=read_data(root/"data")
    known=defaultdict(set);vgroup=defaultdict(set)
    bytail=defaultdict(list);reltrain=defaultdict(list)
    for h,r,t in train:
        h,r,t=int(h),int(r),int(t)
        known[(h,r)].add(t);bytail[t].append((h,r,t));reltrain[r].append((h,t))
    for h,r,t in valid:vgroup[(int(h),int(r))].add(int(t))
    cases=json.loads((root/"data"/f"phase22_seed{args.seed}.json").read_text(encoding='utf-8'))["rows"][12:39]
    assert len(cases)==27
    results=[];beg=time.perf_counter()
    for idx,oldcase in enumerate(cases[:args.limit]):
        h,rel=oldcase["query"]
        targets=sorted(vgroup[(h,rel)]-known[(h,rel)])[:3]
        assert len(targets)==3,(h,rel)
        case=(h,rel,targets,sorted(known[(h,rel)]))
        guard,heldother,heldsame=split_guard(train,case,E,R,args.kind,bytail,reltrain)
        variants={}
        for family,parameters in (("S",LAMBDAS),("R",FRACTIONS)):
            for param in parameters:
                t0=time.perf_counter()
                updated=edit(E,R,args.kind,case,guard,param,
                             mode="soft" if family=="S" else "rank")
                m=metrics(E,R,args.kind,updated,case,guard,heldother,heldsame)
                m["elapsed_sec"]=time.perf_counter()-t0
                variants[f"{family}{param:g}"]=m
        selection={}
        for B in BUDGETS:
            selection[str(B)]={}
            for family,parameters in (("S",LAMBDAS),("R",FRACTIONS)):
                eligible=[(param,variants[f"{family}{param:g}"]) for param in parameters
                          if variants[f"{family}{param:g}"]["all3"] and
                          variants[f"{family}{param:g}"]["cost_l21"]<=B+1e-8]
                choice=min(eligible,key=lambda x:(x[1]["guard_drift"],x[1]["cost_l21"]))[0] if eligible else None
                selection[str(B)][family]={"selected_parameter":choice,
                    "accepted":int(choice is not None),
                    "selected":variants[f"{family}{choice:g}"] if choice is not None else None}
        results.append({"query":[h,rel],"targets":targets,
                        "n_original_valid_hits":variants["S0"]["original_new_hits"],
                        "n_guard":len(guard),"n_heldother":len(heldother),
                        "n_heldsame":len(heldsame),
                        "variants":variants,"adaptive":selection})
        if (idx+1)%5==0:print("EVAL_PROGRESS",args.kind,args.seed,idx+1,
                             "elapsed",round(time.perf_counter()-beg,1),flush=True)
    output={"kind":args.kind,"seed":args.seed,"epoch":args.epoch,
            "checkpoint_sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest(),
            "n_requests":len(results),"n_full":len(cases),"phase22_fixed_eval_keys":True,
            "TEST_used":False,"label_scope":"VALID new edit targets, TRAIN guard/heldout; no TEST",
            "model_shared_head_tail":True,
            "method_limitation":"Local multi-target adaptation; Rank-Truncated numerical kernel from released KGE Rank Locality source, NOT full author model/cohort/protocol",
            "source_baseline":"phase19/22-inspired old-capacity editing with real shared scoring",
            "budget_actual":"L2,1 row norm, cap B; abstentions count as edit failure",
            "elapsed_seconds":time.perf_counter()-beg,"rows":results}
    out=root/"results"/f"{args.kind.lower()}_seed{args.seed}_ep{args.epoch}_edit_v2.json"
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
    print("EVAL_FINISHED",args.kind,args.seed,round(output["elapsed_seconds"],1),flush=True)
    for label in ("S0","S3","R0.125","R0.015625"):
        l=[row["variants"][label] for row in results]
        print("METHOD",label,"3hits",sum(m["all3"] for m in l),
              "cross",round(float(np.mean([m["cross_damage"] for m in l])),3),
              "cost",round(float(np.mean([m["cost_l21"] for m in l])),3))
    for B in BUDGETS:
        for family in ("S","R"):
            v=[row["adaptive"][str(B)][family] for row in results]
            n=sum(x["accepted"] for x in v)
            print("ADAPTIVE_B",B,family,"accepted",n,
                  "cross_attempt",round(sum(x["selected"]["cross_damage"] for x in v if x["accepted"])/len(v),3),
                  "cost_attempt",round(sum(x["selected"]["cost_l21"] for x in v if x["accepted"])/len(v),3))

if __name__=="__main__":main()