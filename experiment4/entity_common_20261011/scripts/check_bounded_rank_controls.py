"""Scale and independent dual checks before bounded jobs start."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'vendor/legacy_qp'))
    from rank_entity_bounded import bounded_proposal
    from rank_constraints_v2 import solve_min_norm
    rng=np.random.default_rng(20261011);maximum=0.
    for _ in range(256):
        old=rng.normal(size=200);old*=min(1.,3./np.linalg.norm(old));raw=rng.normal(size=200)*1e28
        delta=bounded_proposal(old,raw,3.);norm=np.linalg.norm(old+delta);assert norm<=3.+1e-12;maximum=max(maximum,float(norm))
    A=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]);b=np.array([1.,2.,-100.,-100.]);d,certificate=solve_min_norm(A,b)
    assert certificate['certified'];lengths=np.linalg.norm(A,axis=1);normalized=A/lengths[:,None];rhs=b/lengths
    active=np.abs(normalized@d-rhs)<=1e-5;assert active.sum()==2 and len(rhs)==4
    dual,_=nnls(normalized[active].T,d);stat=np.max(np.abs(normalized[active].T@dual-d));complement=np.max(np.abs(dual*(normalized[active]@d-rhs[active])))
    assert stat<1e-8 and complement<1e-8
    result=dict(status='PASS_BOUNDED_SCALE_AND_ACTIVE_DUAL_CONTROLS',projection_cases=256,max_radius=maximum,
                inactive_dual_rows_checked=2,stationarity=float(stat),complementarity=float(complement),test_read=False)
    out=root/'artifacts/bounded-controls';out.mkdir(parents=True,exist_ok=False);(out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
