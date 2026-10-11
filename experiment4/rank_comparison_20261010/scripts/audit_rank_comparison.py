"""Independent new-vector audit; old QP certificates remain inherited."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.linalg
from run_rank_comparison import sha,verify,load

def main(root,kind):
    start=time.monotonic();bound=verify(root);cfg=json.loads((root/'config.json').read_text())
    out=root/'artifacts'/kind;terminal=json.loads((out/'terminal.json').read_text())
    for n,d in terminal['output_sha256'].items(): assert sha(out/n)==d
    cases,old=load(root,kind)
    with np.load(root/'inputs'/kind/'bundle.npz',allow_pickle=False) as z:
        E=z['entity_embeddings'].astype(np.float64);R=z['relation_embeddings'].astype(np.float64)
        train=z['train_triples'];valid=z['valid_triples']
    with np.load(out/'deltas.npz',allow_pickle=False) as z:
        assert np.array_equal(z['cases'],cases);deltas=z['deltas'].copy()
    records=json.loads((out/'per_case.json').read_text());assert len(records)==len(deltas)==3600
    index={(v['case_index'],v['method']):(v,d) for v,d in zip(records,deltas)}
    assert len(index)==len(records)
    ts=set(map(tuple,train));external=np.asarray([v for v in valid if tuple(v) not in ts])
    def queries(h,r):
        a=E[h];b=R[r]
        if kind=='distmult':return a*b
        a0,a1=np.split(a,2,axis=-1);b0,b1=np.split(b,2,axis=-1)
        return np.concatenate((a0*b0-a1*b1,a0*b1+a1*b0),axis=-1)
    cache={}
    def panel(facts,r,label):
        key=(label,r)
        if key in cache:return cache[key]
        rel=facts[facts[:,1]==r];kept=[];qq=[];ranks=[];truth=[]
        for off in range(0,len(rel),256):
            block=rel[off:off+256];heads,inv=np.unique(block[:,0],return_inverse=True)
            Q=queries(heads,np.full(len(heads),r));S=Q@E.T
            gold=S[inv,block[:,2]];rank=1+np.count_nonzero(S[inv]>gold[:,None],axis=1)
            ids=np.where(rank<=10)[0]
            if len(kept)+len(ids)>5000:ids=ids[:5000-len(kept)]
            kept.extend(block[ids]);qq.extend(Q[inv[ids]]);ranks.extend(rank[ids]);truth.extend(gold[ids])
            if len(kept)>=5000:break
        F=np.asarray(kept,dtype=np.int64).reshape(-1,3);Q=np.asarray(qq).reshape(-1,E.shape[1])
        value=(F,Q,np.asarray(ranks),np.asarray(truth));cache[key]=value;return value
    checks=0;samples=0;maxerr=0.;maxnormres=0.
    # Independent tie boundary: strict rank counts ties as not ahead.
    fixture=np.array([4.]*9+[3.,3.,2.]);assert 1+sum(x>fixture[9] for x in fixture)==10
    assert 1+sum(x>2. for x in fixture)==12;checks+=2
    for ci,(h,r,t) in enumerate(cases):
        h,r,t=map(int,(h,r,t));q=queries(np.array([h]),np.array([r]))[0];score=q@E.T
        gap=max(0.,float(np.partition(score,len(score)-10)[-10])-float(score[t])+1e-6)
        F,C,before,gold=panel(train,r,'train');VF,VC,VB,VG=panel(external,r,'valid')
        if len(C):
            _,s,vh=scipy.linalg.svd(C,full_matrices=False,lapack_driver='gesvd');B=vh[s>1e-6]
        else:B=np.empty((0,E.shape[1]))
        direction=q-B.T@(B@q);den=float(q@direction)
        null=np.zeros_like(q) if gap<=0 or abs(den)<1e-12 else gap/den*direction
        lam=cfg['ridge'][kind]
        X=np.vstack((q,np.sqrt(lam)*C,np.sqrt(1e-8)*np.eye(len(q))))
        y=np.r_[gap,np.zeros(len(C)+len(q))]
        ridge=scipy.linalg.lstsq(X,y,cond=None,lapack_driver='gelsd')[0]
        for name,wanted in [('StrictNull',null),('PublishedRidge',ridge)]:
            row,d=index[(ci,name)]
            err=float(np.linalg.norm(d-wanted)/(1+np.linalg.norm(d)));maxerr=max(maxerr,err)
            assert err<1e-5, (ci,name,err)
            target=float(q@(E[t]+d));rank=1+int(np.count_nonzero(score[np.arange(len(E))!=t]>target))
            assert rank==row['rank_after'];checks+=2
            for facts,Q,br,tr,label in [(F,C,before,gold,'train'),(VF,VC,VB,VG,'valid')]:
                orig=Q@E[t];new=Q@(E[t]+d);ar=br.copy()
                for j,f in enumerate(facts):
                    if f[2]!=t:ar[j]=br[j]+int(new[j]>tr[j])-int(orig[j]>tr[j])
                    else:
                        sc=Q[j]@E.T;ar[j]=1+np.count_nonzero(sc[np.arange(len(E))!=t]>new[j])
                vector=br+(new>tr).astype(int)-(orig>tr).astype(int)
                mask=facts[:,2]!=t;assert np.array_equal(ar[mask],vector[mask])
                damage=int(np.count_nonzero(ar>10))
                assert damage==row['damage_count' if label=='train' else 'external_valid_damage']
                assert len(facts)==row['protected_n' if label=='train' else 'external_valid_n']
                checks+=len(facts)+2
                if ci%20==0:
                    for j in np.unique(np.r_[np.arange(min(2,len(facts))),np.flatnonzero(facts[:,2]==t)[:2]]):
                        full=Q[j]@E.T;full[t]=float(Q[j]@(E[t]+d));rk=1+sum(x>full[facts[j,2]] for x in full.tolist())
                        assert rk==ar[j],(ci,name,label,j,rk,ar[j]);samples+=1
            assert row['safe_success']==int(row['rank_after']<=10 and row['damage_count']==0);checks+=1
            if name=='PublishedRidge':
                residual=float(np.linalg.norm(X.T@(X@d-y))/(1+np.linalg.norm(X)*np.linalg.norm(y)))
                maxnormres=max(maxnormres,residual);assert residual<1e-7;checks+=1
        # Exact old vectors were reused, and all old metrics replayed by producer.
        for (i,name),(row,d) in old.items():
            if i==ci:
                actual,vec=index[(i,name)];assert np.array_equal(vec,d)
                assert actual['case_id']==row['case_id'];checks+=2
        if (ci+1)%100==0:print('AUDIT_PROGRESS',kind,ci+1,flush=True)
    verify(root)
    result=dict(status='PASS_SUPPLEMENTARY_BASELINES_AND_REUSED_VECTOR_REPLAY',model=kind,checks=checks,
        discrepancies=0,new_baseline_vectors_audited=800,old_vectors_byte_identical=2800,
        direct_full_candidate_scalar_rank_checks=samples,max_relative_vector_error=maxerr,
        max_ridge_scaled_stationarity=maxnormres,elapsed_s=time.monotonic()-start,
        source_sha256=sha(__file__),input_sha256=bound,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        limitations=['Old QP feasibility/KKT certificates are inherited, not independently re-solved in this round.',
          'New baseline full candidate score rescoring is deterministic sampled; all protected rank changes are checked by exact one-column algebra.',
          'Exposed VALID cohort; frozen-query column contract; all-split Support has extra label inputs.'])
    (out/'audit.json').write_text(json.dumps(result,indent=2));print('AUDIT_PASS',kind,checks,samples,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True)
    a=p.parse_args();main(a.root.resolve(),a.model)
