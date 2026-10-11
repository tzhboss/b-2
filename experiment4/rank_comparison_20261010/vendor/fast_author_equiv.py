"""Semantics-preserving candidate-tail locality acceleration.

Original unmodified kge_rank_locality CLI is the authority; this adapter
keeps its exact solvers and methods but caches all relation protection and
uses the mathematical consequence that ONLY ONE candidate tail score changes.

Acceptance: compare ALL per-case/method metrics with official CLI on
public LibKGE DistMult+ComplEx 256d same 2 validation cases before use.
This is NOT identical source and must be labeled validated-accelerated.
"""
from __future__ import annotations
import csv,hashlib,sys,collections,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str((ROOT.parent/"phase6"/"kge-rank-locality"/"src").resolve()))
from kge_rank_locality import KGEEmbeddings,load_bundle
from kge_rank_locality.core import (
    ProtectionSubspace,DEFAULT_FRACTIONS,DEFAULT_LAMBDAS,
    promotion_score_gap,rank_from_scores,rank_truncated_delta,
    ridge_preservation_delta,SweepResult,LocalityResult
)

class FastFrontier:
    def __init__(self,model,train,top_k=10,max_protected=5000):
        self.model=model
        self.train=np.asarray(train,dtype=np.int64)
        self.top_k=top_k;self.max_protected=max_protected
        self.by_relation={}
        self.stats={"preprocessing_seconds":0.,"cache_hits":0,"computed_relations":0}

    def _build(self,rel):
        if rel in self.by_relation:
            self.stats["cache_hits"]+=1
            return self.by_relation[rel]
        start=time.monotonic()
        rel_rows=self.train[self.train[:,1]==int(rel)]
        selected=[];qrows=[];ranks=[];true_scores=[];bheads=[]
        for startidx in range(0,len(rel_rows),256):
            part=rel_rows[startidx:startidx+256]
            heads,inv=np.unique(part[:,0],return_inverse=True)
            Q=self.model.queries(heads,np.full(len(heads),int(rel),dtype=np.int64))
            Sc=Q@self.model.entity.T
            # Each triple shares the score matrix of its head; correct
            # scores retrieved by contiguous row index.
            gold=Sc[inv,part[:,2]]
            ranked=1+np.count_nonzero(Sc[inv]>gold[:,None],axis=1)
            for i,rank in enumerate(ranked):
                if rank<=self.top_k:
                    selected.append(part[i].copy())
                    qrows.append(Q[inv[i]].copy())
                    ranks.append(int(rank))
                    true_scores.append(float(gold[i]))
                    bheads.append(int(part[i,0]))
                    if self.max_protected>0 and len(selected)>=self.max_protected:break
            if self.max_protected>0 and len(selected)>=self.max_protected:break
        protected=np.asarray(selected,dtype=np.int64).reshape(-1,3)
        Q=np.asarray(qrows,dtype=np.float64).reshape(-1,self.model.dimension)
        sp=ProtectionSubspace.from_directions(Q,dimension=self.model.dimension)
        self.by_relation[rel]=(protected,sp,np.asarray(ranks,dtype=np.int64),
            np.asarray(true_scores,dtype=np.float64),np.asarray(bheads,dtype=np.int64))
        self.stats["computed_relations"]+=1
        self.stats["preprocessing_seconds"]+=time.monotonic()-start
        return self.by_relation[rel]

    def _locality(self,rel,tail,delta,entry):
        protected,sp,old_ranks,true_scores,_=entry
        if not len(protected):
            empty=np.empty(0,dtype=np.int64)
            return LocalityResult(0,0,empty,empty)
        E=self.model.entity
        qs=sp.directions
        # Dot products use the same float64 values as the author package.
        orig_competitor=qs@E[int(tail)]
        edit_tail=E[int(tail)]+delta
        new_competitor=qs@edit_tail
        trutails=protected[:,2]
        after=old_ranks.copy()
        other=trutails!=tail
        if other.any():
            after[other]+=((new_competitor[other]>true_scores[other]).astype(np.int64)
                           -(orig_competitor[other]>true_scores[other]).astype(np.int64))
        self_selected=np.where(~other)[0]
        for j in self_selected:
            head,r,t=map(int,protected[j])
            base=self.model.scores(head,r)
            # target itself changes while every other candidate stays put;
            # ranking must account for all competing scores vs edited target.
            threshold=float(qs[j]@edit_tail)
            ahead=np.count_nonzero(np.concatenate((base[:tail],base[tail+1:]))>threshold)
            after[j]=int(1+ahead)
        dmg=int(np.count_nonzero((old_ranks<=self.top_k)&(after>self.top_k)))
        return LocalityResult(int(np.count_nonzero(old_ranks<=self.top_k)),dmg,old_ranks,after)

    def run_one(self,edit_triple):
        h,rel,tail=map(int,edit_triple)
        entry=self._build(rel)
        _,sp,_,_,_=entry
        score=self.model.scores(h,rel)
        rank_orig=rank_from_scores(score,tail)
        gap=promotion_score_gap(score,tail,top_k=self.top_k,margin=1e-6)
        q=self.model.query(h,rel)
        out=[]
        for f in DEFAULT_FRACTIONS:
            delta,req,used=rank_truncated_delta(q,gap,sp,f)
            post=score.copy();post[tail]=float(q@(self.model.entity[tail]+delta))
            rank=rank_from_scores(post,tail)
            locality=self._locality(rel,tail,delta,entry)
            residual=sp.directions@delta
            out.append(SweepResult("rank_truncated",float(f),rank_orig,rank,
                rank<=self.top_k,locality,float(np.linalg.norm(delta)),
                float(np.max(np.abs(residual))) if residual.size else 0.,
                req,used))
        for lam in DEFAULT_LAMBDAS:
            delta=ridge_preservation_delta(q,gap,sp,lam)
            post=score.copy();post[tail]=float(q@(self.model.entity[tail]+delta))
            rank=rank_from_scores(post,tail)
            locality=self._locality(rel,tail,delta,entry)
            residual=sp.directions@delta
            out.append(SweepResult("ridge",float(lam),rank_orig,rank,
                rank<=self.top_k,locality,float(np.linalg.norm(delta)),
                float(np.max(np.abs(residual))) if residual.size else 0.,None,None))
        return out

def score_csv_input(cases):
    with Path(cases).open("r",newline="",encoding="utf-8") as f:
        return [tuple(int(r[x]) for x in ("h_id","r_id","t_id")) for r in csv.DictReader(f)]

def read_benchmark(bundle_path,case_path):
    data=load_bundle(bundle_path)
    ff=FastFrontier(data.model,data.train_triples)
    records=[]
    for i,case in enumerate(score_csv_input(case_path)):
        result=ff.run_one(case)
        key=":".join(map(str,case))
        for x in result:
            records.append({"case_index":i,"case_id":key,"h_id":case[0],"r_id":case[1],"t_id":case[2],
               "method":x.method,"parameter":x.parameter,
               "rank_before":x.rank_before,"rank_after":x.rank_after,
               "success_at_k":int(x.success),
               "protected_n":x.locality.protected_count,
               "damage_count":x.locality.damage_count,
               "no_damage":int(x.locality.no_damage),
               "safe_success":int(x.safe_success),
               "update_norm":x.update_norm,"residual_max_abs":x.residual_max_abs,
               "requested_rank":x.requested_rank,"used_rank":x.used_rank})
    return records,ff.stats
