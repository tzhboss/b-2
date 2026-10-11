"""Phase25 controlled protection selection / actual-cost matched multi-target KGE edit.

Evaluate exactly phase24 3-model-seed × 27 queries for 5/25 epoch,
DistMult and ComplEx. TRAIN/VALID only. Query and heldout regenerated
with frozen phase24 protocol; all final ranks score shared head/tail params.
No official TEST or publisher full SOTA result used.

Five regimes:
Direct; phase24 legacy tail-only lambda3; exact shared-entity Jacobian
on ALL incident TRAIN guard facts; top-2 incident Jacobians; two
unrelated TRAIN guard pseudo-directions with IDENTICAL constraint count.
Within each candidate family λ=0.3,1,3 predeclared; candidates chosen
ONLY using TRAIN guard drift, not heldout. Exact budget matching:
C(candidate) <= min(5.2, C(Direct) + tolerance) PER EXACT EDIT.
Direct fallback if no candidate feasible. Editing 3 goals jointly by
phase24 slot-protection protocol; not author full KGEditor/SOTA.

One major caveat: 'random pseudo' is a negative control, not a meaningful
real shared-parameter gradient of unrelated facts (which is exactly zero).
"""
import argparse,hashlib,json,time,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
for module_dir in (Path(__file__).resolve().parent.parent.parent/"code", Path(__file__).resolve().parent.parent/"phase24"):
    if (module_dir/"eval_paired_kge_edit.py").exists():
        sys.path.insert(0,str(module_dir))
        break
import eval_paired_kge_edit as b
LAMBDAS=(0.3,1.,3.)
B=5.2
HEADROOMS=(0.,0.2,0.5,1.0)
REGIMES=("legacy_tailonly","incident_all","incident_top2","nonincident_pseudo2")
def gradient(e,r,kind,triple,entity):
    """Exact d score(h,rel,t)/d shared entity row for DistMult/ComplEx.

    Both h==entity and t==entity contribute (when h==t, both).
    """
    h,rel,t=map(int,triple)
    out=np.zeros(e.shape[1],dtype=np.float64)
    if t==entity:out+=b.qvec(e,r,h,rel,kind)
    if h==entity:
        if kind=="DistMult":
            out+=r[rel]*e[t]
        else:
            d=e.shape[1]//2
            rr,ri=r[rel,:d],r[rel,d:]
            tr,ti=e[t,:d],e[t,d:]
            out+=np.concatenate([rr*tr+ri*ti,-ri*tr+rr*ti])
    return out

def get_directions(e,r,kind,guard,entity,mode,random_seed):
    incident=[]
    unrelated=[]
    for tr in guard:
        h,rr,t=map(int,tr)
        if h==entity or t==entity:
            g=gradient(e,r,kind,tr,entity)
            if np.linalg.norm(g)>1e-10:
                # Sort by gradient magnitude, TRAIN guard only.
                incident.append((float(np.linalg.norm(g)),tuple(map(int,tr)),g))
        else:
            unrelated.append(tuple(map(int,tr)))
    if mode=="incident_all":
        vals=[v[2] for v in incident]
    elif mode=="incident_top2":
        vals=[v[2] for v in sorted(incident,key=lambda a:(-a[0],a[1]))[:2]]
    elif mode=="nonincident_pseudo2":
        # Negative-control *pseudo-directions*, NOT true shared gradients
        # (the true gradient for nonincident triples is identically 0).
        # Match the incident_top2 direction COUNT exactly per changed entity.
        k=min(2,len(incident))
        rng=np.random.default_rng(random_seed)
        if unrelated:
            choices=rng.choice(len(unrelated),size=k,replace=len(unrelated)<k)
            vals=[b.qvec(e,r,unrelated[int(ix)][0],unrelated[int(ix)][1],kind) for ix in choices]
        else:
            # Explicit random-vector fallback: no nonincident TRAIN guards
            # for this entity. Keep matching count, record as negative control.
            vals=[rng.normal(size=e.shape[1]) for _ in range(k)]
    else:
        raise ValueError(mode)
    if not vals:
        return np.empty((0,e.shape[1])),len(incident),0
    arr=np.asarray(vals,dtype=np.float64)
    arr/=np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-12)
    return arr,len(incident),len(vals)

def edited(e,r,kind,case,guard,lam,mode,seed):
    if mode=="legacy_tailonly":
        z=b.edit(e,r,kind,case,guard,lam)
        return z,{"directions_per_edited_row":[]}
    h,rel,targets,known=case
    first=b.scores(e,r,h,rel,kind)
    non=np.asarray([i for i in range(len(e)) if i not in set(targets)],np.int32)
    old=[int(t) for t in known if t not in targets and b.top10(first,t)]
    old=sorted(old,key=lambda t:first[t],reverse=True)
    best=[t for t in old if int(np.count_nonzero(first[non]>first[t]))<7]
    chosen=best+[t for t in old if t not in best][:max(0,7-len(best))]
    q=b.qvec(e,r,h,rel,kind);qs=float(q@q)
    out=e.copy()
    if qs<1e-10:return out,{"directions_per_edited_row":[]}
    cache={};counts=[]
    def advance(t,threshold):
        gap=float(threshold-q@out[t])
        if gap<=1e-10:return
        if t not in cache:
            salt=int(hashlib.sha256((str(seed)+":"+str(h)+":"+str(rel)+":"+str(t)).encode()).hexdigest()[:14],16)
            G,n_inc,n_dir=get_directions(e,r,kind,guard,t,mode,salt)
            if len(G):
                vec=np.linalg.solve(np.eye(len(q))+lam*(G.T@G),q)
                den=float(q@vec)
                v=vec/den if den>1e-10 else q/qs
            else:v=q/qs
            cache[t]=v
            counts.append({"entity":int(t),"incident":int(n_inc),"used_directions":int(n_dir)})
        out[t]+=gap*cache[t]
    if chosen:
        disallowed=np.setdiff1d(non,np.asarray(chosen,dtype=int),assume_unique=True)
        spare=7-len(chosen)
        threshold=float(np.partition(first[disallowed],-(spare+1))[-(spare+1)])+1e-5
        for t in chosen:advance(t,threshold)
    actual=b.scores(out,r,h,rel,kind)
    threshold=float(np.partition(actual[non],-8)[-8])+1e-5
    for t in targets:advance(t,threshold)
    return out,{"directions_per_edited_row":counts}

def run(kind,seed,epoch,work,limit):
    start=time.perf_counter()
    data=work/"data"
    train,valid=b.read_data(data)
    ckpath=work/"checkpoints"/f"{kind.lower()}_seed{seed}_epoch{epoch}.pt"
    cp=torch.load(ckpath,map_location="cpu",weights_only=True)
    E,R=b.representations(cp)
    groups=defaultdict(set);known=defaultdict(set)
    bytail=defaultdict(list);byrel=defaultdict(list)
    for hh,rr,t in train:
        hh,rr,t=map(int,(hh,rr,t))
        known[(hh,rr)].add(t);bytail[t].append((hh,rr,t));byrel[rr].append((hh,t))
    for hh,rr,t in valid:groups[(int(hh),int(rr))].add(int(t))
    prior=json.loads((data/f"phase22_seed{seed}.json").read_text())["rows"][12:39]
    earlier=json.loads((work/"results"/f"{kind.lower()}_seed{seed}_ep{epoch}_edit.json").read_text())["rows"]
    assert len(prior)==len(earlier)==27
    rows=[]
    for i,(case_ref,old) in enumerate(zip(prior,earlier)):
        if i>=limit:break
        h,rel=case_ref["query"]
        assert [h,rel]==old["query"]
        target=sorted(groups[(h,rel)]-known[(h,rel)])[:3]
        assert target==old["targets"]
        case=(h,rel,target,sorted(known[(h,rel)]))
        guard,heldother,heldsame=b.split_guard(train,case,E,R,kind,bytail,byrel)
        assert len(guard)==old["n_guard"] and len(heldother)==old["n_heldother"]
        direct=b.edit(E,R,kind,case,guard,0.)
        dm=b.metrics(E,R,kind,direct,case,guard,heldother,heldsame)
        # same scorer and same checkpoint must reproduce phase24 Direct exactly
        for key in ("cross_damage","same_damage","old_damage","all3","cost_l21"):
            assert abs(dm[key]-old["variants"]["0"][key])<1e-7,(kind,seed,epoch,i,key,dm[key],old["variants"]["0"][key])
        candidates={"Direct":{"lambda":0.,"metrics":dm,"direction_counts":[]}}
        legacy=b.edit(E,R,kind,case,guard,3.)
        lm=b.metrics(E,R,kind,legacy,case,guard,heldother,heldsame)
        assert abs(lm["cost_l21"]-old["variants"]["3"]["cost_l21"])<1e-7
        candidates["legacy_tailonly_lam3"]={"lambda":3.,"metrics":lm,"direction_counts":[]}
        for reg in ("incident_all","incident_top2","nonincident_pseudo2"):
            for lam in LAMBDAS:
                updated,info=edited(E,R,kind,case,guard,lam,reg,seed)
                met=b.metrics(E,R,kind,updated,case,guard,heldother,heldsame)
                candidates[f"{reg}_lam{lam:g}"]={"lambda":lam,"metrics":met,
                    "direction_counts":info["directions_per_edited_row"]}
                if reg in ("incident_top2","nonincident_pseudo2") and lam==LAMBDAS[0]:
                    # Same per-tail direction COUNT for fair local/unrelated controls
                    other="nonincident_pseudo2" if reg=="incident_top2" else "incident_top2"
                    if f"{other}_lam{lam:g}" in candidates:
                        a=candidates[f"{reg}_lam{lam:g}"]["direction_counts"]
                        bb=candidates[f"{other}_lam{lam:g}"]["direction_counts"]
                        assert [z["entity"] for z in a]==[z["entity"] for z in bb]
                        assert [z["used_directions"] for z in a]==[z["used_directions"] for z in bb]
        selections={}
        for H in HEADROOMS:
            cap=min(B,dm["cost_l21"]+H)
            refs={}
            for name in ("incident_all","incident_top2","nonincident_pseudo2","legacy_tailonly"):
                trial=[(key,x) for key,x in candidates.items()
                    if key=="Direct" or key.startswith(name+"_lam") or (name=="legacy_tailonly" and key=="legacy_tailonly_lam3")]
                # All methods retain Direct fallback, ensuring equal success
                direct_feasible=bool(dm["all3"] and dm["cost_l21"]<=B+1e-8)
                viable=[(key,x) for key,x in trial if direct_feasible and x["metrics"]["all3"] and x["metrics"]["cost_l21"]<=cap+1e-8]
                choice=min(viable,key=lambda a:(a[1]["metrics"]["guard_drift"],a[1]["metrics"]["cost_l21"],a[0])) if viable else None
                refs[name]={"candidate":choice[0] if choice else None,
                    "metric":choice[1]["metrics"] if choice else None}
                if choice:
                    assert choice[1]["metrics"]["cost_l21"]<=cap+1e-8
            selections[str(H)]={"cap":cap,"direct_feasible":bool(dm["all3"] and dm["cost_l21"]<=B+1e-8),"methods":refs}
        rows.append({"seed":seed,"query":[h,rel],"targets":target,"n_guard":len(guard),
            "n_cross_holdout":len(heldother),"n_same_holdout":len(heldsame),
            "candidates":candidates,"matched":selections})
        if (i+1)%9==0:print("PROGRESS",kind,seed,epoch,i+1,"seconds",round(time.perf_counter()-start,1),flush=True)
    result={"kind":kind,"seed":seed,"epoch":epoch,"n_eval":len(rows),
       "checkpoint_sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest(),
       "protocol":"phase24 exact fixed queries, TRAIN guard separate TRAIN heldout, VALID edit targets, no TEST; actual shared head/tail score; ranked top10; 4 per-case headrooms relative to actual Direct L21, hard max B5.2",
       "global_hard_budget":B,"headrooms":list(HEADROOMS),"lambdas":list(LAMBDAS),
       "selection_label_scope":"TRAIN guard score drift ONLY; TRAIN holdout is evaluation only",
       "author_full_sota":False,"official_test_read":False,
       "elapsed_s":time.perf_counter()-start,"rows":rows}
    outdir=work/"phase25"/"results";outdir.mkdir(parents=True,exist_ok=True)
    dest=outdir/f"{kind.lower()}_seed{seed}_ep{epoch}_matched.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("SAVED",dest,"sec",round(result["elapsed_s"],1),flush=True)
    for H in HEADROOMS:
        v=[x["matched"][str(H)]["methods"] for x in rows]
        for name in ("incident_all","incident_top2","nonincident_pseudo2","legacy_tailonly"):
            a=[x[name]["metric"] for x in v]
            print("RESULT",kind,seed,epoch,H,name,
                  "accepted",sum(z is not None for z in a),
                  "non_direct",sum(x[name]["candidate"] not in ("Direct",None) for x in v),
                  "damage",round(float(np.mean([z["cross_damage"] if z else 0 for z in a])),4),
                  "cost",round(float(np.mean([z["cost_l21"] if z else 0 for z in a])),4),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--work",type=Path,default=Path("."))
    parser.add_argument("--kind",choices=("DistMult","ComplEx"),required=True)
    parser.add_argument("--seed",type=int,choices=(401,402,403),required=True)
    parser.add_argument("--epoch",type=int,choices=(5,25),default=25)
    parser.add_argument("--limit",type=int,default=27)
    a=parser.parse_args()
    run(a.kind,a.seed,a.epoch,a.work,a.limit)
