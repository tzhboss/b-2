"""Original full-step audit with corrected active dual indexing and separate outputs."""
import json
import numpy as np
import torch
from pathlib import Path
from common_training import sha
from rank_entity import cached

def audit_rank_steps(root,cfg,job):
    """Replay every accepted QP's original full linear system independently."""
    from scipy.optimize import nnls
    torch.set_num_threads(2);out=root/'artifacts/incremental'/job['id'];t=json.loads((out/'training-receipt.json').read_text())
    f=cached(root/cfg['feature_directory']/f"s{job['seed']}"/'guards.npz');z=cached(out/'training-replay.npz')
    G=f['true_Q'];P=f['pred_Q'];tg=f['true_gold'];pg=f['pred_cut'];pq=f['pred_queries']
    tc=z['initial_true_counts'].copy();pc=z['initial_pred_counts'].copy();rows=json.loads((out/'steps.json').read_text());checks=0;accepted=0;failures=[];maximum_violation=0.;worst_kkt=0.
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True);E=initial['E'].numpy().copy();R=initial['R'].numpy()
    for i,row in enumerate(rows):
        epoch,index,h,rel,tail=map(int,z['ids'][i]);old=z['old_vectors'][i];new=z['new_vectors'][i];d=new-old
        assert np.array_equal(E[tail],old),'Shared entity replay continuity'
        assert np.array_equal(z['q'][i],E[h]*R[rel]),'Updated head participates in true scoring'
        oldt=G@old;newt=G@new;oldp=P@old;newp=P@new
        mask=(tc==9)&(oldt<=tg);pmask=(pc==9)&(oldp<=pg)&~((pq[:,0]==h)&(pq[:,1]==rel))
        nt=tc+(newt>tg).astype(np.int64)-(oldt>tg).astype(np.int64);np_=pc+(newp>pg).astype(np.int64)-(oldp>pg).astype(np.int64)
        E[tail]=new;checks+=3
        if row['qp_used']:
            accepted+=1;A=[z['q'][i][None,:],-G[mask]];b=[np.array([row['gap']]),oldt[mask]-tg[mask]+cfg['guard_margin']]
            if job['method']=='QPPredRankNovel':A.append(-P[pmask]);b.append(oldp[pmask]-pg[pmask]+cfg['guard_margin'])
            A=np.vstack(A);b=np.concatenate(b);slack=A@d-b
            violation=max(0.,float(-slack.min()));maximum_violation=max(maximum_violation,violation)
            assert violation<=cfg['qp_primal_tolerance'] and np.linalg.norm(d)<=row['rank_norm_cap']+cfg['norm_cap_tolerance']
            assert not np.any((tc<=9)&(nt>9))
            if job['method']=='QPPredRankNovel':assert not np.any((pc<=9)&(np_>9)&~((pq[:,0]==h)&(pq[:,1]==rel)))
            score=(E[h]*R[rel])@E.T;assert 1+np.count_nonzero(score>score[tail])<=10
            # The native solver reports full KKT diagnostics. Reconstruct an
            # independent nonnegative dual witness from the accepted vector.
            lengths=np.linalg.norm(A,axis=1);nz=lengths>0;a=A[nz]/lengths[nz,None];rhs=b[nz]/lengths[nz]
            active=np.abs(a@d-rhs)<=cfg['audit_active_tolerance']
            if active.any():
                dual,residual=nnls(a[active].T,d,maxiter=cfg['audit_nnls_maxiter']);stat=float(np.max(np.abs(a[active].T@dual-d)))
                complement=float(np.max(np.abs(dual*(a[active]@d-rhs[active]))))
                assert stat<=cfg['audit_kkt_tolerance'] and complement<=cfg['audit_kkt_tolerance']
                worst_kkt=max(worst_kkt,stat,complement)
            else:assert np.linalg.norm(d)<=cfg['audit_kkt_tolerance']
            checks+=6
        elif job['method'].startswith('QP'):assert np.array_equal(z['actual_delta'][i],z['rank_delta'][i]);checks+=1
        tc=nt;pc=np_
        if (i+1)%2048==0:print('QP_REPLAY_AUDIT',job['id'],i+1,len(rows),accepted,flush=True)
    assert np.array_equal(tc,z['terminal_true_counts']) and np.array_equal(pc,z['terminal_pred_counts'])
    assert accepted==t['qp_accepted'];assert t['all_original_facts_visited']
    # Every replayed original fact ID must be covered, not just easy edit rows.
    assert set((z['ids'][:,1]//2).tolist())==set(range(t['novel_original_facts']))
    replay_state=dict(status='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS',checks=checks,accepted_qp_checked=accepted,
                      steps_replayed=len(rows),max_original_primal_violation=maximum_violation,worst_independent_kkt_error=worst_kkt,
                      scope='All steps continuity/guard crossing; every accepted QP full primal/cap/target/frontier and reconstructed nonnegative dual',
                      implementation_sha256=sha(Path(__file__)))
    (root/cfg['reaudit_directory']/job['id']/'solver-audit.json').write_text(json.dumps(replay_state,indent=2));print(json.dumps(replay_state))
