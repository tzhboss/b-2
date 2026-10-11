"""Independent low-dimensional active-set oracle and shared-model controls."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np


def oracle(u,A,b,H):
    best=None;best_value=float('inf');dim=len(u)
    for n in range(dim+1):
        for ids in itertools.combinations(range(len(b)),n):
            aa=A[list(ids)];bb=b[list(ids)]
            if n:
                try:
                    z=np.linalg.solve(H,aa.T);lam=np.linalg.solve(aa@z,aa@u-bb);d=u-z@lam
                except np.linalg.LinAlgError:continue
                if (lam < -1e-9).any():continue
            else:d=u.copy()
            if len(b) and (A@d-b>1e-8).any():continue
            value=.5*float((d-u)@H@(d-u))
            if value<best_value:best=d;best_value=value
    assert best is not None;return best


def run(root,cfg):
    sys.path.insert(0,str(root/'src'))
    from task_metric_projection import query_metric,nearest_task_feasible,verify_task_projection
    from safe_proposal import radius_segment
    rng=np.random.default_rng(cfg['seed']);opts=cfg['solver'];cases=[];max_error=0.
    for dim in (2,3):
        for k in range(cfg['cases_per_dimension']):
            q=rng.normal(size=(dim+2,dim));H=query_metric(q,dim)
            A=np.r_[rng.normal(size=(dim+3,dim)),np.eye(dim),-np.eye(dim)];b=rng.uniform(0,2,size=len(A))
            if k%3==0:b[:dim+3]=0.
            u=rng.normal(size=dim)*3;expected=oracle(u,A,b,H)
            actual,c=nearest_task_feasible(u,A,b,H,opts);assert c['certified'],(dim,k,c)
            verify_task_projection(u,A,b,H,actual,c,opts)
            error=float(np.max(np.abs(actual-expected)));assert error<cfg['vector_tolerance'],(dim,k,error);max_error=max(max_error,error)
            eig=np.linalg.eigvalsh(H);assert eig.min()>=1-1e-12 and eig.max()<=dim+1+1e-12
            cases.append(dict(dimension=dim,case=k,max_abs_vector_error=error))
    assert np.array_equal(query_metric(np.zeros((0,3)),3),np.eye(3))
    # Identity metric must preserve the Euclidean optimizer, independent oracle.
    u=np.array([3.,2.]);A=np.array([[1.,0.],[0.,1.]]);b=np.ones(2);H=np.eye(2)
    d,c=nearest_task_feasible(u,A,b,H,opts);assert c['certified'] and np.allclose(d,[1.,1.],atol=1e-12)
    # A rotated metric protects task scores while remaining under the same guard.
    q=np.array([[1.,1.]]);H=query_metric(q,2);b=np.array([0.]);A=np.array([[1.,0.]]);u=np.array([1.,0.])
    d,c=nearest_task_feasible(u,A,b,H,opts);assert c['certified'];assert np.allclose(d,[0.,.5],atol=1e-12)
    verify_task_projection(u,A,b,H,d,c,opts)
    # Verify full-candidate old ranks after cumulative shared-vector updates.
    old=np.column_stack([np.arange(12,0,-1),np.r_[np.arange(1,10),0,-1,-2]]).astype(float)
    E=np.vstack([old,[-1.,-1.],[-.5,-2.]]);R=np.array([[1/12,0.],[0.,1.]])
    directions=E[0]*R;cutoff=directions@E[9]
    counts=np.array([9,9]);radius=float(np.linalg.norm(E,axis=1).max());head_change=False
    for tail,goal in zip([12,13],[[8.,5.],[7.,4.]]):
        before=E[tail].copy();current=directions@before;mask=(counts==9)&(current<=cutoff)
        A=directions[mask];b=cutoff[mask]-current[mask];b-=np.minimum(cfg['guard_margin'],b*.5)
        H=query_metric(np.array([[1.,1.],[2.,-.5]]),2)
        d,c=nearest_task_feasible(np.asarray(goal)-before,A,b,H,opts);assert c['certified'];verify_task_projection(np.asarray(goal)-before,A,b,H,d,c,opts)
        alpha=radius_segment(before,d,radius);E[tail]=before+alpha*d
        counts+=(directions@E[tail]>cutoff).astype(int)-(current>cutoff).astype(int)
        scores=directions@E.T;actual=1+(scores>scores[:,9,None]).sum(1)
        assert np.array_equal(actual,counts+1) and (actual<=10).all()
        head_change |= not np.array_equal((before*R[1])@E.T,(E[tail]*R[1])@E.T)
    assert np.array_equal(E[:12],old) and head_change
    # Forced failure must return zero, never an uncertified learned proposal.
    failopts=dict(opts,maxiter=1);d,c=nearest_task_feasible(np.array([3.,2.]),np.eye(2),np.ones(2),np.eye(2),failopts)
    assert not c['certified'] and np.array_equal(d,np.zeros(2))
    result=dict(passed=True,scope='Synthetic correctness only; not real-graph utility',oracle_cases=len(cases),max_abs_vector_error=max_error,
                identity_metric_control=True,score_direction_control=True,shared_model_old_top10_preserved=True,new_head_uses_shared_update=True,
                frozen_old_vectors=True,zero_fallback_control=True,test_read=False,graph_facts_read=False,cases=cases)
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False);(out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
