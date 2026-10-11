"""Enumerate small-QP optima independently; adversarial and radius controls."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np


def oracle(u,A,b):
    candidates=[]
    for size in range(len(u)+1):
        for active in itertools.combinations(range(len(b)),size):
            if active:
                a=A[list(active)];rhs=a@u-b[list(active)]
                dual=np.linalg.lstsq(a@a.T,rhs,rcond=None)[0]
                if dual.min() < -1e-9:continue
                d=u-a.T@dual
            else:d=u.copy()
            if np.max(A@d-b)<=1e-8:candidates.append((float((d-u)@(d-u)),d))
    assert candidates
    return min(candidates,key=lambda v:v[0])[1]


def main(root,config):
    sys.path.insert(0,str(root/'src'));from safe_proposal import nearest_feasible,radius_segment,verify_projection
    cfg=json.loads((root/config).read_text());opts=cfg['solver'];rng=np.random.default_rng(cfg['synthetic_seed']);cases=[]
    for dim in (2,3):
        for _ in range(cfg['ordinary_cases_per_dimension']):
            A=rng.normal(size=(cfg['synthetic_rows'],dim));b=rng.uniform(0,1,size=len(A));u=rng.normal(size=dim)*2;cases.append((u,A,b))
    cases.extend([(np.array([2.,2.]),np.array([[1.,0.],[0.,1.],[-1.,0.],[1.,0.],[0.,0.]]),np.array([0.,.5,1.,0.,0.])),
                  (np.array([-2.,3.]),np.array([[1.,1.],[-1.,-1.],[1.,0.]]),np.zeros(3))])
    rows=[];max_error=0.;failures=[]
    for index,(u,A,b) in enumerate(cases):
        d,c=nearest_feasible(u,A,b,opts);verified=verify_projection(u,A,b,d,c,opts);expected=oracle(u,A,b)
        error=float(np.linalg.norm(d-expected));max_error=max(max_error,error)
        if not c['certified'] or error>cfg['oracle_vector_tolerance']:failures.append(dict(case=index,certificate=c,oracle_error=error))
        rows.append(dict(case=index,dimension=len(u),oracle_error=error,status=c['status'],certificate_verified=True))
    # Fallback correctness cannot be mistaken for successful optimization.
    forced=dict(opts,maxiter=1);u=np.array([10.,4.]);A=np.eye(2);b=np.ones(2)
    d,c=nearest_feasible(u,A,b,forced);verify_projection(u,A,b,d,c,opts)
    assert not c['certified'] and np.array_equal(d,np.zeros(2))
    radius_max=0.;radius_checks=0
    for _ in range(cfg['radius_cases']):
        old=rng.normal(size=3);old=old/np.linalg.norm(old)*rng.uniform(0,3)
        proposal=rng.normal(size=3)*10**rng.uniform(-5,28);alpha=radius_segment(old,proposal,3.)
        value=float(np.linalg.norm(old+alpha*proposal));radius_max=max(radius_max,value)
        assert 0<=alpha<=1 and value<=3.+cfg['radius_tolerance'];radius_checks+=1
        # Full guard feasibility persists after scaling towards zero update.
        a=rng.normal(size=(5,3));bb=np.maximum(a@proposal,0)+rng.uniform(0,.1,size=5)
        assert np.max(a@(alpha*proposal)-bb)<=cfg['radius_tolerance']
    receipt=dict(check_scope='Synthetic nearest-QP and radius correctness only; no graph facts or metrics read',
                 passed=not failures,ordinary_and_degenerate_cases=len(cases),max_oracle_vector_error=max_error,failures=failures,
                 forced_solver_failure_zero_update=True,radius_cases=radius_checks,max_radius=radius_max,test_read=False,new_train_read=False)
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));(out/'cases.json').write_text(json.dumps(rows,indent=2));print(json.dumps(receipt),flush=True)
    assert not failures,'Synthetic projection did not pass independent oracle controls'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();main(a.root.resolve(),a.config)
