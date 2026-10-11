"""Analytic controls for native solvers and shared-head candidate competition."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();root=a.root.resolve()
    cfg=json.loads((root/a.config).read_text());sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/cfg['qp_vendor']))
    from rank_entity import native_core,guard_counts
    from rank_constraints_v2 import solve_min_norm
    core=native_core(root,cfg);checks=[]
    for A,b,want in [(np.array([[1.,0.],[-1.,1.]]),np.array([1.,0.]),[1.,1.]),
                     (np.array([[1.,0.],[0.,1.],[0.,0.]]),np.array([2.,3.,-1.]),[2.,3.])]:
        d,c=solve_min_norm(A,b);assert c['certified'];np.testing.assert_allclose(d,want,atol=1e-8,rtol=0);checks.append('Independent closed-form min-norm QP')
    d,c=solve_min_norm(np.array([[0.,0.]]),np.array([.1]));assert not c['certified'];checks.append('Exact zero-row infeasible detection')
    d,c=solve_min_norm(np.array([[1.,0.],[-1.,0.]]),np.array([2.,-1.]));assert not c['certified'];checks.append('Opposing promotion/guard rejected')
    sp=core.ProtectionSubspace.from_directions(np.array([[0.,1.]]),dimension=2)
    d,requested,used=core.rank_truncated_delta(np.array([1.,1.]),2.,sp,.5)
    np.testing.assert_allclose(d,[2.,0.],atol=1e-12,rtol=0);assert requested==used==1;checks.append('Native projected Rank analytic geometry')
    E=torch.tensor([[float(i)] for i in range(10,0,-1)]+[[0.],[0.]],dtype=torch.float64)
    G=torch.tensor([[1.]],dtype=torch.float64);truth=torch.tensor([1.],dtype=torch.float64)
    counts=guard_counts(G,truth,E,1);assert counts.item()==9
    for tail,value in [(10,1.5),(11,2.)]:
        old=G@E[tail];new=G@torch.tensor([value],dtype=torch.float64)
        counts+=(new>truth).long()-(old>truth).long();E[tail]=value
        assert torch.equal(counts,guard_counts(G,truth,E,1))
    assert counts.item()==11;checks.append('Two novel competitors displace an immutable old score')
    before_E=E.clone();before_E[10]=0.;q0=before_E[10];q1=E[10]
    s0=q0@before_E.T;s1=q1@E.T
    assert 1+int((s0>s0[9]).sum())==1 and 1+int((s1>s1[9]).sum())==12
    assert not torch.equal(s0[:10],s1[:10]);checks.append('Shared novel head changes old-candidate scores and ranks')
    result=dict(status='PASS_ANALYTIC_AND_SHARED_ENTITY_CONTROLS',checks=len(checks),controls=checks,test_read=False)
    out=root/'artifacts/rank-controls';out.mkdir(parents=True,exist_ok=False);(out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
