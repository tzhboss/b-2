"""Local rank solver independent derivative and whole-model controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch


def run(root,cfg):
    sys.path.insert(0,str(root/'src'))
    from shared_rank_solver import LocalPairProblem,GuardSeparator,solve_shared_update
    from shared_rank_cache_v2 import SharedRankCache
    rng=np.random.default_rng(cfg['seed']);max_error=0.;cases=[]
    for case in range(cfg['jacobian_cases']):
        E=rng.normal(size=(13,3));R=rng.normal(size=(2,3));ids=[0,8,9,12]
        target=E[ids]+rng.normal(size=(4,3))*.2;weights=np.array([4.,4.,4.,1.]);radii=np.full(4,2.)
        p=LocalPairProblem(E,R,ids,target,weights,E[ids],radii,cfg['solver']['guard_margin'])
        p.add([(0,0,9,8),(8,1,8,12),(9,0,0,9),(12,1,12,9)])
        x=target.ravel();J=p.jacobian(x);B=p.ball_jacobian(x);grad=p.gradient(x)
        numeric=np.zeros_like(J);nb=np.zeros_like(B);ng=np.zeros_like(grad)
        for k in range(len(x)):
            d=np.zeros_like(x);d[k]=cfg['difference_step']
            numeric[:,k]=(p.values(x+d)-p.values(x-d))/(2*cfg['difference_step'])
            nb[:,k]=(p.ball_values(x+d)-p.ball_values(x-d))/(2*cfg['difference_step'])
            ng[k]=(p.objective(x+d)-p.objective(x-d))/(2*cfg['difference_step'])
        error=max(float(np.max(np.abs(J-numeric))),float(np.max(np.abs(B-nb))),float(np.max(np.abs(grad-ng))))
        assert error<cfg['derivative_tolerance'],(case,error);max_error=max(max_error,error);cases.append(dict(case=case,error=error))
    old=np.column_stack([np.arange(12,0,-1),np.r_[np.arange(1,10),0,-1,-2]]).astype(float)
    E=np.vstack([old,[-1.,-1.]]);R=np.array([[1/12,0.],[0.,1.]])
    keys=np.array([[0,0],[0,1],[8,0],[9,0]])
    refs=np.array([[0,9]]+[[1,i] for i in range(10)]+[[2,9],[3,9]])
    device=torch.device(cfg['device']);et=torch.as_tensor(E,device=device);rt=torch.as_tensor(R,device=device)
    c=SharedRankCache(et,rt,keys,refs,k=10,score_batch=4)
    ids=[8,12];target=E[ids].copy();target[1]=[5.,-1.]
    patch,info=solve_shared_update(c,ids,target,[4.,1.],E[ids],[2.,7.],[8],cfg['solver'])
    assert info['accepted'],info
    assert info['cuts']>0 and info['cut_rounds']>0
    assert np.array_equal(c.E.cpu().numpy(),E),'Solver mutated cache before commit'
    assert c.commit(patch);c.full_recount();model=c.E.cpu().numpy()
    def rank(E,h,r,g):
        scores=[sum(float(E[h,d]*R[r,d])*float(E[t,d]) for d in range(E.shape[1])) for t in range(len(E))]
        return 1+sum(s>scores[g] for s in scores)
    assert rank(model,0,0,12)<=10 and rank(model,0,0,9)<=10 and rank(model,8,0,9)<=10
    for q,g in refs:assert rank(model,*keys[q],g)<=10
    # Exact analytic toy optimum for the registered old-weight objective.
    m=cfg['solver']['guard_margin'];root_x=(3+np.sqrt(9-48*m))/2
    expected=np.array([[root_x,9.],[5.,-1.]]);analytic_error=float(np.max(np.abs(model[ids]-expected)))
    assert analytic_error<cfg['analytic_tolerance'],(model[ids],expected,info)
    frozen=SharedRankCache(et,rt,keys,refs,k=10,score_batch=4)
    fpatch,finfo=solve_shared_update(frozen,[12],target[1:],[1.],E[[12]],[7.],[],cfg['solver'])
    assert fpatch.passed
    fmodel=E.copy();fmodel[12]=fpatch.rows.cpu().numpy()[0]
    assert rank(fmodel,0,0,12)>10
    # Forced working-set budget returns unchanged shared rows even when target unsafe.
    fail=SharedRankCache(et,rt,keys,refs,k=10,score_batch=4)
    _,failure=solve_shared_update(fail,ids,target,[4.,1.],E[ids],[2.,7.],[8],dict(cfg['solver'],max_cut_rounds=0))
    assert not failure['accepted'] and np.array_equal(fail.E.cpu().numpy(),E)
    # Absolute old-row movement centers are fixed, not reset after each update.
    next_target=target.copy();next_target[0,0]=2.;next_target[1,0]=6.
    bounded_patch,bounded=solve_shared_update(c,ids,next_target,[4.,1.],E[ids],[1.1,8.],[8],cfg['solver'])
    assert bounded['accepted'],bounded
    assert np.linalg.norm(bounded_patch.rows.cpu().numpy()[0]-E[8])<=1.1+cfg['solver']['movement_tolerance']
    assert bounded_patch.rows.cpu().numpy()[0,0]>=2.9-1e-8
    result=dict(passed=True,scope='Synthetic local cutting-plane implementation only; no graph utility',
                jacobian_cases=len(cases),max_derivative_error=max_error,analytic_toy_vector_error=analytic_error,
                new_gold_rank=rank(model,0,0,12),all_reference_ranks_safe=True,old_head_actual_rescore=True,
                solver_did_not_commit_before_acceptance=True,full_rank_cache_replay=True,
                absolute_movement_centers_checked=True,forced_cut_budget_returns_unchanged=True,
                joint=info,frozen_control=finfo,test_read=False,graph_facts_read=False,cases=cases)
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'synthetic_models.npz',initial=E,relations=R,joint=model,keys=keys,refs=refs)
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
