"""CPU all-candidate sample inference and independent full transition checks."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_scoring import load_tables,sha
from rank_metrics import transitions

def audit(base,root,spec):
    start=time.monotonic();torch.set_num_threads(2)
    tables,queries,bound=load_tables(base,spec);out=root/'artifacts'/spec['id'];term=json.loads((out/'terminal.json').read_text())
    for n,d in term['output_sha256'].items():assert sha(out/n)==d
    # Independent NumPy normalization/city-block reduction; target read from full score row.
    normalized=[tuple(x.numpy()/np.linalg.norm(x.numpy(),axis=1,keepdims=True) for x in pair) for pair in tables]
    checked=0;maxerr=0.;rank_checks=0;allvalues={}
    for cohort in range(5):
        with np.load(out/f'cohort{cohort}.npz',allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        assert np.array_equal(v['queries'],queries[cohort]);limit=len(tables[cohort][0]);n=len(queries[cohort])
        inds=np.unique(np.linspace(0,n-1,min(64,n),dtype=int))
        for prefix,table,pool in [('before',cohort,limit),('after',4,limit),('expanded',4,len(tables[4][0]))]:
            E,R=normalized[table]
            for ix in inds:
                h,r,t=map(int,queries[cohort][ix]);dist=np.abs(E[h]+R[r]-E[:pool]).sum(axis=1)
                scores=1/(1+np.exp(-(9-dist)));gold=float(scores[t]);rank=1+sum(x>gold for x in scores.tolist())
                err=abs(gold-float(v[prefix+'_gold'][ix]));maxerr=max(maxerr,err)
                assert err<=1e-12,(spec['id'],cohort,prefix,ix,err)
                assert rank==v[prefix+'_rank'][ix],(spec['id'],cohort,prefix,ix,rank,v[prefix+'_rank'][ix])
                mask=np.arange(pool)!=t;boundary=np.partition(scores[mask],-10)[-10]
                assert abs(boundary-v[prefix+'_boundary'][ix])<=1e-12
                rank_checks+=1
        assert np.all(v['expanded_rank']>=v['after_rank']);checked+=n
        if cohort<4:
            actual=transitions(v['before_rank'],v['after_rank'],v['before_gold'],v['after_gold'],v['before_std'])
            assert actual==term['cohorts'][cohort]['fixed'];checked+=n
            expansion=transitions(v['after_rank'],v['expanded_rank'],v['after_gold'],v['expanded_gold'],v['after_std'])
            assert expansion==term['cohorts'][cohort]['expansion_only']
            for key,value in v.items():
                if key!='queries':allvalues.setdefault(key,[]).append(value)
        else:assert np.mean(v['expanded_rank']<=10)==term['summary']['new_hits10']
        print('AUDIT_PROGRESS',spec['id'],cohort,flush=True)
    av={k:np.concatenate(v) for k,v in allvalues.items()}
    overall=transitions(av['before_rank'],av['after_rank'],av['before_gold'],av['after_gold'],av['before_std'])
    assert all(term['summary'][k]==v for k,v in overall.items())
    expansion=transitions(av['after_rank'],av['expanded_rank'],av['after_gold'],av['after_gold'],np.ones(len(av['after_rank'])))
    assert all(term['expansion'][k]==v for k,v in expansion.items())
    for n,d in bound.items():assert sha(base/n)==d
    report=dict(status='PASS_RAW_STRICT_RANK_REEVALUATION' if not spec.get('exploratory') else 'PASS_NUMERICS_UPSTREAM_INTEGRITY_FAIL_RETAINED',
      id=spec['id'],full_query_derivation_checks=checked,sampled_full_candidate_rank_checks=rank_checks,max_target_score_error=maxerr,
      failures=[],source_sha256=sha(__file__),bound_input_sha256=bound,terminal_sha256=sha(out/'terminal.json'),elapsed_s=time.monotonic()-start,
      limits=['All rank aggregates checked; independent checkpoint score inference is sampled64/cohort, not independent full score replay.',
      'Native checkpoints/data/query sets and training/model differences retained; this is a common metric framework, not a controlled SOTA ranking.'])
    (out/'audit.json').write_text(json.dumps(report,indent=2));print('AUDIT_PASS',spec['id'],rank_checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();spec=next(x for x in json.loads((a.root/'config.json').read_text())['jobs'] if x['id']==a.job);audit(a.base.resolve(),a.root.resolve(),spec)
