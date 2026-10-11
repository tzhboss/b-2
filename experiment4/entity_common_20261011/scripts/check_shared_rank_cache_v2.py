"""Independent group-equivalence and full-pool cache-patch controls."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch


def scalar_scores(E,R,h,r):
    # Independent CPU scalar sum; no producer scorer or grouping reduction.
    return [sum(float(E[h,d]*R[r,d])*float(E[c,d]) for d in range(E.shape[1])) for c in range(len(E))]


def run(root,cfg):
    sys.path.insert(0,str(root/'src'))
    from shared_rank_cache_v2 import SharedRankCache,fixed_scores
    started=time.monotonic();torch.set_num_threads(2)
    device=torch.device(cfg['device']);rng=np.random.default_rng(cfg['seed'])
    passed=0;rejected=0;max_error=0.;cases=[]
    for case in range(cfg['cases']):
        E=rng.normal(size=(cfg['entities'],cfg['dimension']));R=rng.normal(size=(3,cfg['dimension']))
        keys=np.array([(h,r) for h in [0,1,8,9,12] for r in range(3)])
        refs=[]
        for qi,(h,r) in enumerate(keys):
            s=scalar_scores(E,R,h,r);order=sorted(range(len(E)),key=lambda c:(-s[c],c))
            refs.extend((qi,g) for g in order[:cfg['k']])
        et=torch.as_tensor(E,dtype=torch.float64,device=device);rt=torch.as_tensor(R,dtype=torch.float64,device=device)
        cache=SharedRankCache(et,rt,keys,refs,k=cfg['k'],score_batch=cfg['score_batch'])
        original=cache.E.clone();original_scores=cache.S.clone();original_q=cache.Q.clone();original_version=cache.version
        ids=np.array([0,8,12,20]);rows=E[ids]+rng.normal(size=(len(ids),E.shape[1]))*(0.005 if case%2==0 else 1.)
        # Verify full-vs-subset fixed arithmetic, across different batch sizes.
        q=cache.Q;full=fixed_scores(q,cache.E)
        assert torch.equal(full[:,ids],fixed_scores(q,cache.E[ids]))
        assert torch.equal(full,torch.cat([fixed_scores(part,cache.E) for part in q.split(1)]))
        patch=cache.propose(ids,rows);candidate=E.copy();candidate[ids]=rows
        expected=[];all_member_ranks=[]
        for qi,(h,r) in enumerate(keys):
            s=scalar_scores(candidate,R,h,r);members=[g for key,g in refs if key==qi]
            member_ranks=[1+sum(v>s[g] for v in s) for g in members]
            threshold=min(s[g] for g in members)
            weakest_rank=1+sum(v>threshold for v in s)
            assert weakest_rank==max(member_ranks)
            expected.append(weakest_rank-1);all_member_ranks.extend(member_ranks)
        assert np.array_equal(patch.counts.cpu().numpy(),expected), (case,patch.counts,expected)
        assert patch.passed==(max(all_member_ranks)<=cfg['k'])
        assert torch.equal(cache.E,original) and torch.equal(cache.S,original_scores) and torch.equal(cache.Q,original_q)
        assert cache.version==original_version
        accepted=cache.commit(patch);assert accepted==patch.passed
        if accepted:
            passed+=1;cache.full_recount()
            assert np.array_equal(cache.E.cpu().numpy(),candidate)
            # A previously committed patch is stale even if its rows are safe.
            try:cache.commit(patch)
            except AssertionError:pass
            else:raise AssertionError('Stale proposal accepted')
        else:
            rejected+=1;assert torch.equal(cache.E,original) and torch.equal(cache.S,original_scores)
            cache.full_recount()
        # Compare current score values to independently evaluated scalar scores.
        current=cache.E.cpu().numpy()
        for qi,(h,r) in enumerate(keys):
            error=float(np.max(np.abs(cache.S[qi].cpu().numpy()-scalar_scores(current,R,h,r))))
            max_error=max(max_error,error);assert error<cfg['score_tolerance']
        # Exact unchanged patch always commits, then a full independent pool check.
        unchanged=cache.propose([9],cache.E[[9]].clone());assert unchanged.passed and cache.commit(unchanged);cache.full_recount()
        cases.append(dict(case=case,accepted=accepted,full_rescored_queries=patch.rescored_queries))
    # Strict-score tie convention and changing identity of weakest reference.
    E=np.array([[1.,0.],[2.,0.],[1.,0.],[0.,1.],[-1.,0.]])
    R=np.array([[1.,0.]]);keys=np.array([[0,0]]);refs=np.array([[0,0],[0,1],[0,2]])
    et=torch.as_tensor(E,dtype=torch.float64,device=device);rt=torch.as_tensor(R,dtype=torch.float64,device=device)
    c=SharedRankCache(et,rt,keys,refs,k=3,score_batch=cfg['score_batch'])
    assert c.counts.item()==1
    p=c.propose([1],np.array([[.5,0.]]));assert p.passed and p.counts.item()==2
    assert c.commit(p);c.full_recount();assert c.thresholds.item()==.5
    unsafe=c.propose([1],np.array([[-.5,0.]]));assert not unsafe.passed and not c.commit(unsafe);c.full_recount()
    result=dict(passed=True,scope='Scoring/cache implementation controls only; not graph method utility',device=str(device),
                cases=len(cases),accepted_proposals=passed,rejected_proposals=rejected,max_scalar_score_error=max_error,
                exact_full_subset_and_batch_scores=True,all_reference_group_equivalence=True,
                actual_changed_heads_golds_and_candidates=True,proposal_is_read_only=True,rejection_retains_state=True,
                stale_patch_rejected=True,full_recount_after_every_commit=True,strict_ties_checked=True,
                changing_weakest_reference_checked=True,graph_facts_read=False,test_read=False,
                elapsed_s=time.monotonic()-started,cases_detail=cases)
    assert passed>0 and rejected>0
    output=root/cfg['output_directory'];output.mkdir(parents=True,exist_ok=False)
    (output/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases_detail'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
