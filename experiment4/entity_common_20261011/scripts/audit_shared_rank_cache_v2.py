"""Independent scalar all-member audit on new randomized shared patches."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import yaml


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root,cfg):
    p=root/'experiments'/cfg['id']/'experiment.yaml';state=yaml.safe_load(p.read_text())
    assert state['status']=='completed_pending_independent_check'
    for name,digest in state['frozen_input_sha256'].items():assert sha(root/name)==digest
    receipt_path=root/cfg['output_directory']/'receipt.json';receipt=json.loads(receipt_path.read_text())
    assert receipt['passed'] and not receipt['test_read'] and not receipt['graph_facts_read']
    assert receipt['cases']==cfg['cases'] and len(receipt['cases_detail'])==cfg['cases']
    assert receipt['accepted_proposals']>0 and receipt['rejected_proposals']>0
    assert receipt['max_scalar_score_error']<=cfg['score_tolerance']
    sys.path.insert(0,str(root/'src'));from shared_rank_cache_v2 import SharedRankCache,fixed_scores
    torch.set_num_threads(2);device=torch.device(cfg['device']);rng=np.random.default_rng(cfg['seed']+11007)
    def scores(E,R,h,r):
        return [sum(float(E[h,k])*float(R[r,k])*float(E[t,k]) for k in range(E.shape[1])) for t in range(len(E))]
    checked_members=0;safe=0;unsafe=0;max_error=0.
    for case in range(32):
        E=rng.normal(size=(cfg['entities'],cfg['dimension']));R=rng.normal(size=(3,cfg['dimension']))
        keys=np.array([[h,r] for h in (0,2,4,7) for r in range(3)])
        references=[]
        for qi,(h,r) in enumerate(keys):
            s=scores(E,R,h,r);references.extend((qi,g) for g in sorted(range(len(E)),key=lambda t:(-s[t],t))[:cfg['k']])
        C=SharedRankCache(torch.as_tensor(E,device=device),torch.as_tensor(R,device=device),keys,references,cfg['k'],cfg['score_batch'])
        ids=np.array([0,2,9,12]);rows=E[ids]+rng.normal(size=(len(ids),E.shape[1]))*(0.002 if case%2==0 else 1.)
        proposal=C.propose(ids,rows);candidate=E.copy();candidate[ids]=rows
        scalar=[]
        for qi,(h,r) in enumerate(keys):
            s=scores(candidate,R,h,r)
            members=[g for q,g in references if q==qi]
            ranks=[1+sum(v>s[g] for v in s) for g in members]
            checked_members+=len(ranks);scalar.append(max(ranks)-1)
        assert proposal.counts.cpu().tolist()==scalar
        is_safe=all(n<cfg['k'] for n in scalar);assert proposal.passed==is_safe
        assert np.array_equal(C.E.cpu().numpy(),E)
        assert C.commit(proposal)==is_safe
        expected=candidate if is_safe else E
        assert np.array_equal(C.E.cpu().numpy(),expected)
        C.full_recount()
        for qi,(h,r) in enumerate(keys):
            actual=fixed_scores(C.Q[[qi]],C.E).cpu().numpy()[0]
            error=float(np.max(np.abs(actual-scores(expected,R,h,r))));max_error=max(max_error,error)
            assert error<cfg['score_tolerance']
        safe+=int(is_safe);unsafe+=int(not is_safe)
    assert safe>0 and unsafe>0
    failures=[]
    for eid in ['KGE-ENTITY-COMMON-SHARED-RANK-CACHE-023','KGE-ENTITY-COMMON-SHARED-RANK-CACHE-023R2']:
        previous=yaml.safe_load((root/'experiments'/eid/'experiment.yaml').read_text())
        assert previous['status']=='failed_synthetic_cache_controls_retained' and previous['exit_code']==1
        for name,digest in previous['frozen_input_sha256'].items():assert sha(root/name)==digest
        failures.append(eid)
    audit=dict(status='PASS_NEW_RANDOM_CASES_INDEPENDENT_SCALAR_ALL_MEMBER_RANKS',new_random_cases=32,
               reference_members_checked=checked_members,safe_patches=safe,unsafe_patches=unsafe,
               max_scalar_score_error=max_error,exact_cache_full_rescore=True,failures=[],
               retained_failed_attempts=failures,producer_receipt_sha256=sha(receipt_path),
               implementation_sha256=sha(Path(__file__)),
               limits=['Synthetic cache implementation only; no graph gain or global optimizer guarantee.',
                       'GPU guard arithmetic differs from prior GEMM; common evaluation remains unchanged.'])
    (receipt_path.parent/'audit.json').write_text(json.dumps(audit,indent=2))
    formal=root/'results'/cfg['id'];formal.mkdir(parents=True,exist_ok=False)
    (formal/'summary.json').write_text(json.dumps(dict(result=receipt,audit=audit),indent=2))
    state.update(status='completed_audited_synthetic_cache_controls_only',result_directory=str(formal.relative_to(root)))
    p.write_text(yaml.safe_dump(state,sort_keys=False));print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
