"""Re-evaluate audited saved Rank/QP vectors; no optimizer or training."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_scoring import sha
from rank_metrics import transitions

def evaluate(root,base,spec):
    source=base.parent/'rank_comparison_20261010';kind=spec['kind'];folder=source/'artifacts'/kind
    sys.path.insert(0,str(source/'vendor'))
    from kge_rank_locality.core import KGEEmbeddings
    from fast_author_equiv import FastFrontier
    audit=json.loads((folder/'audit-v2.json').read_text());assert audit['strictnull_discrepancies']==0
    bound={str(folder/'audit-v2.json'):sha(folder/'audit-v2.json')}
    for name in ['deltas.npz','per_case.json','terminal.json']:
        p=folder/name;assert sha(p)==audit['output_sha256'][name];bound[str(p)]=sha(p)
    bundle=source/'inputs'/kind/'bundle.npz';key=str(bundle.relative_to(source))
    assert sha(bundle)==audit['input_sha256'][key];bound[str(bundle)]=sha(bundle)
    for rel in ['vendor/fast_author_equiv.py','vendor/kge_rank_locality/core.py']:
        p=source/rel;assert sha(p)==audit['input_sha256'][rel];bound[str(p)]=sha(p)
    with np.load(bundle,allow_pickle=False) as z:
        model=KGEEmbeddings(z['entity_embeddings'],z['relation_embeddings'],str(z['model']));train=z['train_triples']
    with np.load(folder/'deltas.npz',allow_pickle=False) as z:cases=z['cases'].copy();deltas=z['deltas'].copy()
    rows=json.loads((folder/'per_case.json').read_text());assert len(rows)==len(deltas)==3600
    ff=FastFrontier(model,train);std_cache={};out=root/'artifacts'/spec['id'];out.mkdir(parents=True,exist_ok=False)
    acc={};success={};safe={};sample_checks=0;start=time.monotonic()
    for row,d in zip(rows,deltas):
        if row['method']=='PublishedRidge':continue
        assert time.monotonic()-start<7200
        ci=row['case_index'];h,r,t=map(int,cases[ci]);assert row['case_id']==f'{h}:{r}:{t}'
        F,sp,br,gold,_=entry=ff._build(r);Q=sp.directions
        if r not in std_cache:
            parts=[]
            for off in range(0,len(Q),128):parts.append((Q[off:off+128]@model.entity.T).std(axis=1))
            std_cache[r]=np.concatenate(parts) if parts else np.empty(0)
        loc=ff._locality(r,t,d,entry);ar=loc.ranks_after
        # LocalityResult field names verified against bundled author core.
        after_gold=gold.copy();mask=F[:,2]==t;after_gold[mask]=Q[mask]@(model.entity[t]+d)
        assert loc.damage_count==row['damage_count'] and len(F)==row['protected_n']
        q=model.query(h,r);s=model.scores(h,r);s[t]=q@(model.entity[t]+d)
        erank=1+np.count_nonzero(s>s[t]);assert erank==row['rank_after']
        assert row['safe_success']==int(erank<=10 and loc.damage_count==0)
        name=row['method'];values=acc.setdefault(name,{k:[] for k in ['before_rank','after_rank','before_gold','after_gold','before_std','case_index']})
        for key,v in [('before_rank',br),('after_rank',ar),('before_gold',gold),('after_gold',after_gold),('before_std',std_cache[r]),('case_index',np.full(len(F),ci))]:values[key].append(v)
        success.setdefault(name,[]).append(int(erank<=10));safe.setdefault(name,[]).append(row['safe_success'])
        if ci%20==0:
            ids=np.unique(np.r_[np.arange(min(2,len(F))),np.flatnonzero(mask)[:2]])
            for ix in ids:
                full=Q[ix]@model.entity.T;full[t]=Q[ix]@(model.entity[t]+d)
                rank=1+sum(x>full[F[ix,2]] for x in full.tolist());assert rank==ar[ix];sample_checks+=1
        if ci%100==0 and name=='Direct':print('RANK_PROGRESS',kind,ci,flush=True)
    summaries=[]
    for name,parts in acc.items():
        v={k:np.concatenate(a) for k,a in parts.items()};np.savez_compressed(out/(name+'.npz'),**v)
        summaries.append(dict(method=name,dataset='FB15k237',model=kind,update_unit='400 independent single-tail-column edit requests',
            probe_split='relation TRAIN first5000 initially Top10 facts, repeated across edits',candidate_contract='same_full_entity_candidates',
            scope='frozen head query; one tail score column changes',evidence_status='PENDING_AUDIT',new_hits10=float(np.mean(success[name])),safe_success=float(np.mean(safe[name])),
            extra_all_split_labels='AllSplit' in name,**transitions(v['before_rank'],v['after_rank'],v['before_gold'],v['after_gold'],v['before_std'])))
    for p,d in bound.items():assert sha(p)==d
    term=dict(id=spec['id'],status='TERMINAL_PENDING_AUDIT',summaries=summaries,bound_input_sha256=bound,
        output_sha256={p.name:sha(p) for p in out.glob('*.npz')},sampled_full_candidate_scalar_rank_checks=sample_checks,
        elapsed_s=time.monotonic()-start,implementation_sha256=sha(__file__),ridge_disposition='Prior vector audit FAIL retained; excluded from promoted comparison',
        limitations=['Exposed VALID, native same400 cases; inherited QP certificates.', 'TRAIN protection scope and frozen query differ from snapshot learning and KGEditor batches.'])
    (out/'terminal.json').write_text(json.dumps(term,indent=2));print('RANK_TERMINAL',kind,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();spec=next(x for x in json.loads((a.root/'config.json').read_text())['jobs'] if x['id']==a.job);evaluate(a.root,a.base,spec)
