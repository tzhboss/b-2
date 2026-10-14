"""Boundary routing with local conflict and learning-negative controls."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,'/data/lc/untzh/probability67_20261014')
import experiment67 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;prev.prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 **{f'boundary_{geometry}_{weight}':dict(route='boundary',geometry=geometry,signal='absolute',weight=weight)
     for geometry in ['entity','coordinate'] for weight in [1.,1.1,1.2]},
 **{f'weight{weight}_{signal}':dict(route='uniform',geometry='none',signal=signal,weight=weight)
     for weight in [1.1,1.2] for signal in ['none','absolute']},
 **{f'filter_{signal}':dict(route='uniform',geometry='none',signal=signal,weight=1.,filtered=True)
     for signal in ['none','absolute','mass_sample']},
 'mass_entity':dict(route='uniform',geometry='entity',signal='mass_sample',weight=1.),
 'mass_coordinate':dict(route='uniform',geometry='coordinate',signal='mass_sample',weight=1.),
 'boundary_mass_entity':dict(route='boundary',geometry='entity',signal='mass_sample',weight=1.1),
}


def learning(E,R,new,tape,j,spec,data):
    lo,hi=tape['bounds'][j:j+2]
    p,nh,nt=[torch.as_tensor(a,device='cuda') for a in (new[tape['pos'][lo:hi]],tape['nh'][lo:hi],tape['nt'][lo:hi])]
    h,r,t=p.unbind(1);ps=(E[h]*R[r]*E[t]).sum(-1);ns=(E[nh]*R[r[:,None]]*E[nt]).sum(-1)
    negative=F.softplus(ns)
    if spec.get('filtered'):
        legal=data['negative_legal'][j]
        nl=(negative*legal).sum(-1)/legal.sum(-1).clamp_min(1)
    else:nl=negative.mean(-1)
    weights=torch.where((h<ex.N)&(t>=ex.N),spec['weight'],1.)
    return ((F.softplus(-ps)+nl)*weights).mean()+.0005*(E[h].square().mean()+R[r].square().mean())


def gradient_geometry(gp,gl,kind):
    if kind=='entity':return core.entity_project(gp,gl)[0]
    if kind=='coordinate':return torch.where(gp*gl>=0,gp,0.)
    return gp


def calibrate(initial,prefix,new,tape,data):
    report={};rng=np.random.default_rng(17);selections=[rng.choice(ex.SIZE,32,replace=False) for _ in range(8)]
    for name,spec in ARMS.items():
        if spec['signal']=='none':continue
        E,R,opt=core.model(initial,prefix);rows=[]
        for idx,j in enumerate(range(370,378)):
            loss=learning(E,R,new,tape,j,spec,data);gl=torch.autograd.grad(loss,E)[0]
            p=prev.penalty(E,data,selections[idx],spec['signal']);gp=torch.autograd.grad(p,E)[0]
            rows.append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
        a=np.array(rows);denom=np.sqrt(np.mean(a[:,1]**2))
        report[name]={'lambda':.1*np.sqrt(np.mean(a[:,0]**2))/denom if denom else 0.,'norms':rows}
    ex.dump('calibration.json',report);return report


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    boundary=np.argsort(-data['oldrank'],kind='stable')[:int(np.ceil(.2*ex.SIZE))]
    for j in range(370,740):
        selected=rng.choice(ex.SIZE,32,replace=False) if spec['route']=='uniform' else np.concatenate([rng.choice(boundary,26,replace=False),rng.choice(ex.SIZE,6,replace=False)])
        loss=learning(E,R,new,tape,j,spec,data);gl=torch.autograd.grad(loss,E)[0]
        p=E.new_zeros(());alpha=0.;gp=torch.zeros_like(E)
        if spec['signal']!='none':
            p=prev.penalty(E,data,selected,spec['signal']);gp=torch.autograd.grad(p,E)[0]
            gp=gradient_geometry(gp,gl,spec['geometry'])
            bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
            alpha=min(cal[name]['lambda'],.1*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha})
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
    tick=time.monotonic();data=prev.prepare(initial,old,new,tape)
    facts=set(map(tuple,np.concatenate([old,new])));legal={};bad=0
    for j in range(370,740):
        lo,hi=tape['bounds'][j:j+2];p=new[tape['pos'][lo:hi]]
        nh,nt=tape['nh'][lo:hi],tape['nt'][lo:hi]
        a=np.array([[(int(h),int(r),int(t)) not in facts for h,t in zip(hs,ts)]
                    for r,hs,ts in zip(p[:,1],nh,nt)],dtype=bool)
        bad+=int((~a).sum());legal[j]=torch.as_tensor(a,device='cuda')
    data['negative_legal']=legal
    ex.dump('negative-preparation.json',{'known_positive_negative_draws':bad,'total_draws':sum(v.numel() for v in legal.values()),'source':'TRAIN facts only'})
    cal=calibrate(initial,prefix,new,tape,data);costs={'prepare_seconds':time.monotonic()-tick,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
