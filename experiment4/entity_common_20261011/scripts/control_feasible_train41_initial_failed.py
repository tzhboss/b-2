"""Additional active-constraint and unchanged-fallback controls for full run."""
import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from feasible_train_optimization import optimize_train_row
from shared_rank_cache_v2 import SharedRankCache
from control_feasible_train40 import controls as prior_controls


def controls():
    prior=prior_controls()
    E=torch.tensor([[1.,0.],[.5,1.],[2.,0.],[1.,0.],[0.,0.]],dtype=torch.float64)
    R=torch.ones((1,2),dtype=torch.float64)
    class Index:
        def examples(self,cache,entity,count):return np.array([[1,0,4]]),np.array([[1,0,2]]),None,None
    options=dict(guard_margin=1e-8,ftol=1e-10,maxiter=200,max_cut_rounds=8,cut_rows_per_round=64,
                 primal_tolerance=1e-10,movement_tolerance=1e-8)
    cache=SharedRankCache(E,R,[[0,0]],[[0,3]],k=2,score_batch=2)
    patch,receipt,p,n=optimize_train_row(cache,Index(),4,np.zeros(2),3.,.0005,10,options)
    assert receipt['accepted'] and receipt['history'][0]['violating_queries']==1
    assert len(receipt['history'])>1 and receipt['history'][-1]['cuts']>=1
    assert receipt['after_loss']<receipt['before_loss']-.1 and patch.rows[0,0]<=1.-1e-8+1e-10
    assert cache.commit(patch);cache.full_recount()
    final=cache.E.numpy();s=np.array([sum(final[0]*R.numpy()[0]*row) for row in final])
    assert 1+int((s>s[3]).sum())==2
    # An exhausted cut budget must return the unchanged safe row, not the
    # unconstrained optimizer result or a silently relaxed guard.
    cache2=SharedRankCache(E,R,[[0,0]],[[0,3]],k=2,score_batch=2)
    fallback,fr,_,_=optimize_train_row(cache2,Index(),4,np.zeros(2),3.,.0005,10,dict(options,max_cut_rounds=0))
    assert not fr['accepted'] and np.array_equal(fallback.rows.numpy(),E[[4]].numpy()) and fallback.passed
    assert torch.equal(cache2.E,E) and cache2.version==0
    return dict(status='PASS_ACTIVE_RANK_CUT_TRAIN_DESCENT_AND_UNCHANGED_BUDGET_FALLBACK',
                prior_controls=prior,active_cut_rounds=len(receipt['history'])-1,
                constrained_loss_before=receipt['before_loss'],constrained_loss_after=receipt['after_loss'],
                guard_rank=2,scientific_utility_asserted=False)

if __name__=='__main__':print(json.dumps(controls(),indent=2))
