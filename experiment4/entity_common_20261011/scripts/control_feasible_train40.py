"""Independent scalar/gradient/feasible toy controls, never utility evidence."""
import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from feasible_train_optimization import TrainLossProblem,score_polynomial,optimize_train_row
from shared_rank_cache_v2 import SharedRankCache


def controls():
    rng=np.random.default_rng(409);score_checks=gradient_checks=0;worst=0.
    for case in range(48):
        E=rng.normal(size=(7,4));R=rng.normal(size=(3,4));entity=6;x=rng.normal(size=4)
        p=np.array([[6,0,1],[2,1,6],[6,2,6],[1,0,2]])
        n=np.array([[6,0,3],[2,1,4],[6,2,5],[1,0,6]])
        problem=TrainLossProblem(E,R,entity,p,n,E[entity],10.,.0005,1e-8)
        actual=E.copy();actual[entity]=x
        for triples,poly in ((p,problem.pos),(n,problem.neg)):
            direct=np.array([sum(actual[h]*R[r]*actual[t]) for h,r,t in triples])
            assert np.allclose(direct,problem.score(x,poly),rtol=1e-12,atol=1e-12)
            score_checks+=len(triples)
        pos=np.array([sum(actual[h]*R[r]*actual[t]) for h,r,t in p])
        neg=np.array([sum(actual[h]*R[r]*actual[t]) for h,r,t in n])
        expected=np.logaddexp(0,-pos).mean()+np.logaddexp(0,neg).mean()+.0005*np.mean(x*x)
        assert abs(expected-problem.objective(x))<1e-12
        eps=1e-6;numeric=[]
        for j in range(4):
            d=np.zeros(4);d[j]=eps
            numeric.append((problem.objective(x+d)-problem.objective(x-d))/(2*eps))
        err=float(np.max(np.abs(problem.gradient(x)-numeric)));worst=max(worst,err)
        assert err<2e-8;gradient_checks+=4
    # Feasible row has a free TRAIN direction orthogonal to a saturated guard.
    # Full pool: rank2 gold remains rank2 while TRAIN positive score improves.
    E=torch.tensor([[1.,0.],[0.,1.],[2.,0.],[1.,0.],[0.,0.]],dtype=torch.float64)
    R=torch.ones((1,2),dtype=torch.float64)
    cache=SharedRankCache(E,R,[[0,0]],[[0,3]],k=3,score_batch=2)
    class Index:
        def examples(self,cache,entity,negative_count):
            return np.array([[1,0,4]]),np.array([[1,0,2]]),None,None
    options=dict(guard_margin=1e-8,ftol=1e-10,maxiter=200,max_cut_rounds=8,
                 cut_rows_per_round=64,primal_tolerance=1e-10,movement_tolerance=1e-8)
    patch,receipt,p,n=optimize_train_row(cache,Index(),4,np.zeros(2),3.,.0005,10,options)
    assert receipt['accepted'] and receipt['after_loss']<receipt['before_loss']-.1
    assert cache.commit(patch);cache.full_recount()
    final=cache.E.numpy();scores=np.array([sum(final[0]*R.numpy()[0]*row) for row in final])
    assert 1+int((scores>scores[3]).sum())<=3 and final[4,1]>1.
    return dict(status='PASS_SCALAR_ALIAS_GRADIENT_AND_TRAIN_IMPROVEMENT_WITH_FULL_GUARDS',
                scalar_score_checks=score_checks,gradient_components_checked=gradient_checks,
                max_gradient_error=worst,toy_before_loss=receipt['before_loss'],toy_after_loss=receipt['after_loss'],
                graph_utility_asserted=False)

if __name__=='__main__':print(json.dumps(controls(),indent=2))
