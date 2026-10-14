"""Matched protection-strength controls for TRAIN semantic gating."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,'/data/lc/untzh/search64_20261013')
import experiment64 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 **{f'{method}_{int(rho*10):02d}':dict(kind=kind,route=route,rho=rho)
     for method,kind,route in [('uniform','absolute','uniform'),('relation','relation_exclude','uniform'),
                              ('query_soft','query_soft','uniform'),('query_oldonly','absolute','old_only')]
     for rho in [.2,.3]},
 **{f'combined_{int(rho*10):02d}':dict(kind='combined',route='uniform',rho=rho) for rho in [.1,.2,.3]},
 'oldonly_coordinate_02':dict(kind='absolute',route='old_only',rho=.2,geometry='coordinate'),
 'oldonly_coordinate_03':dict(kind='absolute',route='old_only',rho=.3,geometry='coordinate'),
}
original_penalty=prev.penalty


def penalty(E,data,selected,kind):
    if kind!='combined':return original_penalty(E,data,selected,kind)
    q=data['q_t'][selected];ids=data['candidates_t'][selected]
    r=data['relation_t'][selected]
    rc=data['relation_count_t'][r[:,None],ids-ex.N]
    weight=(rc==0).double()*torch.where(data['query_exposed_t'][selected],.1,1.)[:,None]
    scores=(q[:,None]*E[ids].double()).sum(-1)
    return (weight*torch.nn.functional.softplus(scores)).mean()


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    rho=spec['rho']
    for j in range(370,740):
        pool=data['old_only'] if spec['route']=='old_only' else np.arange(ex.SIZE)
        selected=rng.choice(pool,32,replace=False)
        loss=ex.base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p=penalty(E,data,selected,spec['kind']);gp=torch.autograd.grad(p,E)[0]
        if 'geometry' in spec:gp=prev.geometry(gp,gl,E,spec['geometry'])
        bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
        alpha=min(cal[spec['kind']]['lambda']*rho/.1,rho*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(E).all() or not torch.isfinite(loss):raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),
                        'alpha':alpha,'gradient_ratio':alpha*pn/bn if bn else 0.})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'frozen_rows_equal':True}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();data=prev.enrich(core.prepare_data(initial,old,new),new,len(initial['R']))
    prev.ARMS=ARMS;prev.penalty=penalty
    cal=prev.calibration(initial,prefix,new,tape,data)
    costs={'prepare_seconds':time.monotonic()-tick,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal)
        ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs)
    ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
