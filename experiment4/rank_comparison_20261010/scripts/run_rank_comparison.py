"""Config-driven fixed-cohort replay and supplementary released baselines."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
    manifest=json.loads((root/'input-sha256.json').read_text())
    for name,digest in manifest.items():
        assert sha(root/name)==digest, name
    return manifest

def audit_hashes(value):
    if isinstance(value,dict):
        for v in value.values(): yield from audit_hashes(v)
    elif isinstance(value,list):
        for v in value: yield from audit_hashes(v)
    elif isinstance(value,str) and len(value)==64: yield value

def load(root,kind):
    old={}; cases=None
    for suffix,number in [('snapshot-verification400','034'),('true-ablation400','036'),('all-support400','037')]:
        folder=root/'inputs'/kind/(kind+'-'+suffix)
        audit=json.loads((root/'inputs'/('audit'+number+'.json')).read_text())
        assert audit['status']=='PASS' and audit['discrepancies']==0
        bound=set(audit_hashes(audit))
        for p in folder.iterdir():
            if p.is_file(): assert sha(p) in bound, str(p)
        with np.load(folder/'deltas.npz',allow_pickle=False) as z:
            ds=z['deltas'].copy(); cc=z['cases'].copy()
        if cases is None: cases=cc
        else: assert np.array_equal(cc,cases)
        records=json.loads((folder/'per_case.json').read_text())
        assert len(records)==len(ds)
        for row,d in zip(records,ds):
            i=row['case_index']; assert row['case_id']==':'.join(map(str,cases[i]))
            key=(i,row['method']); assert key not in old
            old[key]=(row,d)
        assert sha(root/'inputs'/kind/'bundle.npz') in bound
    assert cases.shape==(400,3)
    return cases,old

def main(root,kind):
    bound=verify(root);cfg=json.loads((root/'config.json').read_text())
    sys.path.insert(0,str(root/'vendor'))
    from kge_rank_locality.core import KGEEmbeddings,rank_truncated_delta,ridge_preservation_delta,promotion_score_gap,rank_from_scores
    from fast_author_equiv import FastFrontier
    cases,old=load(root,kind)
    selection=json.loads((root/'inputs'/kind/'frozen_selection.json').read_text())
    lam=float(selection['methods_selected']['ridge']['selected_parameter'])
    assert lam==cfg['ridge'][kind]
    with np.load(root/'inputs'/kind/'bundle.npz',allow_pickle=False) as z:
        m=KGEEmbeddings(z['entity_embeddings'],z['relation_embeddings'],str(z['model']))
        train=z['train_triples'];valid=z['valid_triples']
    ts=set(map(tuple,train));external=np.asarray([v for v in valid if tuple(v) not in ts])
    tf=FastFrontier(m,train);vf=FastFrontier(m,external)
    out=root/'artifacts'/kind;out.mkdir(parents=True,exist_ok=False)
    rows=[];vectors=[];start=time.monotonic();replayed=0
    for ci,(h,r,t) in enumerate(cases):
        assert time.monotonic()-start<cfg['budget_seconds_per_model']
        h,r,t=map(int,(h,r,t));q=m.query(h,r);s=m.scores(h,r);entry=tf._build(r);ext=vf._build(r)
        assert rank_from_scores(s,t)>10
        gap=promotion_score_gap(s,t)
        updates={name:d for (i,name),(row,d) in old.items() if i==ci}
        assert set(updates)==set(cfg['reused_methods'])
        updates.update(StrictNull=rank_truncated_delta(q,gap,entry[1],1.)[0],PublishedRidge=ridge_preservation_delta(q,gap,entry[1],lam))
        for name,d in updates.items():
            score=s.copy();score[t]=q@(m.entity[t]+d);rank=rank_from_scores(score,t)
            loc=tf._locality(r,t,d,entry);ex=vf._locality(r,t,d,ext)
            row=dict(model=kind,case_index=ci,case_id=f'{h}:{r}:{t}',relation=r,method=name,
                rank_after=rank,success_at_k=int(rank<=10),no_damage=int(loc.no_damage),
                safe_success=int(rank<=10 and loc.no_damage),damage_count=loc.damage_count,
                protected_n=loc.protected_count,external_valid_damage=ex.damage_count,
                external_valid_n=ex.protected_count,update_norm=float(np.linalg.norm(d)),
                source='reused_audited_vector' if (ci,name) in old else 'new_frozen_baseline')
            if (ci,name) in old:
                ref=old[(ci,name)][0]
                for key in ['rank_after','success_at_k','no_damage','safe_success','damage_count','protected_n','external_valid_damage','external_valid_n']:
                    assert row[key]==ref[key], (kind,ci,name,key,row[key],ref[key])
                replayed+=1
            rows.append(row);vectors.append(d)
        if (ci+1)%50==0: print('PROGRESS',kind,ci+1,round(time.monotonic()-start,2),flush=True)
    (out/'per_case.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(out/'deltas.npz',cases=cases,deltas=np.asarray(vectors))
    summaries=[]
    base=[v for v in rows if v['method']=='PublishedRank']
    for name in cfg['reused_methods']+cfg['new_methods']:
        rr=[v for v in rows if v['method']==name];assert len(rr)==400
        delta=[v['safe_success']-b['safe_success'] for v,b in zip(rr,base)]
        summaries.append(dict(model=kind,method=name,N=len(rr),
            **{k:float(np.mean([v[k] for v in rr])) for k in ['success_at_k','no_damage','safe_success','damage_count','external_valid_damage','update_norm']},
            paired_safe_gain_cases=sum(v==1 for v in delta),paired_safe_loss_cases=sum(v==-1 for v in delta)))
    verify(root)
    result=dict(status='TERMINAL_PENDING_AUDIT',model=kind,summaries=summaries,replayed_old_vectors=replayed,
        elapsed_s=time.monotonic()-start,numpy=np.__version__,source_sha256=sha(__file__),bound_input_sha256=bound,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},new_training=False,new_QP_optimization=False)
    (out/'terminal.json').write_text(json.dumps(result,indent=2));print(json.dumps(summaries),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',choices=['distmult','complex'],required=True)
    a=p.parse_args();main(a.root.resolve(),a.model)
