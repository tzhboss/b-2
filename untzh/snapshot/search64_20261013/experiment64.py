"""TRAIN semantic interference and update geometry; same cached host trajectory."""
import collections
import copy
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F

SOURCE63=Path('/data/lc/untzh/search63_20261013')
sys.path.insert(0,str(SOURCE63))
import experiment63 as core
ex=core.ex
ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={
 'relation_exclude128':dict(kind='relation_exclude'),
 'relation_soft128':dict(kind='relation_soft'),
 'relation_exclude_full':dict(kind='relation_exclude_full'),
 'query_old_only':dict(kind='absolute',route='old_only'),
 'query_soft':dict(kind='query_soft'),
 'entity_frequency':dict(kind='entity_frequency'),
 'relation_frequency':dict(kind='relation_frequency'),
 'tangent':dict(kind='absolute',geometry='tangent'),
 'radial':dict(kind='absolute',geometry='radial'),
 'coord_nonconflict':dict(kind='absolute',geometry='coordinate'),
 'semantic_tangent':dict(kind='relation_soft',geometry='tangent'),
 'semantic_coordinate':dict(kind='relation_soft',geometry='coordinate'),
 'postadam_02':dict(kind='absolute',postadam=True,rho=.2),
 'postadam_03':dict(kind='absolute',postadam=True,rho=.3),
 'postadam_semantic':dict(kind='relation_soft',postadam=True,rho=.1),
}


def enrich(data,new,relations):
    counts=np.zeros((relations,ex.TOTAL-ex.N),dtype=np.int64)
    mask=new[:,2]>=ex.N
    np.add.at(counts,(new[mask,1],new[mask,2]-ex.N),1)
    queries={(int(h),int(r)) for h,r,t in new if t>=ex.N}
    exposed=np.array([(int(h),int(r)) in queries for h,r,t in data['queries']])
    entitycount=np.bincount(np.concatenate([new[:,0],new[:,2]]),minlength=ex.TOTAL)
    data['relation_count_t']=torch.as_tensor(counts,device='cuda')
    data['query_exposed_t']=torch.as_tensor(exposed,device='cuda')
    data['old_only']=np.flatnonzero(~exposed)
    data['entity_count_t']=torch.as_tensor(entitycount,device='cuda')
    data['relation_t']=torch.as_tensor(data['queries'][:,1],device='cuda')
    ex.dump('semantic-preparation.json',{'old_only_answers':int((~exposed).sum()),'exposed_answers':int(exposed.sum()),
        'relation_candidate_observations':int((counts>0).sum()),'source':'new TRAIN only; inverse relation IDs retained'})
    return data


def penalty(E,data,selected,kind):
    q=data['q_t'][selected]
    r=data['relation_t'][selected]
    if kind=='relation_exclude_full':
        scores=q@E[ex.N:].double().T
        counts=data['relation_count_t'][r]
        legal=data['legal_t'][selected] & (counts==0)
        values=F.softplus(scores).masked_fill(~legal,0)
        return (values.sum(-1)/legal.sum(-1).clamp_min(1)).mean()
    ids=data['candidates_t'][selected]
    scores=(q[:,None]*E[ids].double()).sum(-1)
    rc=data['relation_count_t'][r[:,None],ids-ex.N]
    weights=torch.ones_like(scores)
    if kind=='relation_exclude':weights=(rc==0).double()
    elif kind=='relation_soft':weights=torch.where(rc>0,.1,1.).double()
    elif kind=='relation_frequency':weights=(1+rc.double()).rsqrt()
    elif kind=='entity_frequency':weights=(1+data['entity_count_t'][ids].double()).rsqrt()
    elif kind=='query_soft':weights=weights*torch.where(data['query_exposed_t'][selected],.1,1.)[:,None]
    elif kind!='absolute':raise ValueError(kind)
    return (weights*F.softplus(scores)).mean()


def geometry(gp,gl,E,kind):
    a,b=gp[ex.N:].double(),gl[ex.N:].double()
    if kind=='coordinate':
        a=torch.where(a*b>=0,a,0.)
    else:
        e=E[ex.N:].detach().double()
        radial=(a*e).sum(-1)[:,None]/e.square().sum(-1).clamp_min(1e-30)[:,None]*e
        a=a-radial if kind=='tangent' else radial
    return torch.cat([torch.zeros_like(gp[:ex.N]),a.to(gp.dtype)])


def calibration(initial,prefix,new,tape,data):
    E,R,opt=core.model(initial,prefix)
    rng=np.random.default_rng(17);rows=collections.defaultdict(list)
    for j in range(370,378):
        selected=rng.choice(ex.SIZE,32,replace=False)
        gl=torch.autograd.grad(ex.base.learning(E,R,new,tape,j),E)[0]
        for kind in sorted({s['kind'] for s in ARMS.values()}):
            p=penalty(E,data,selected,kind)
            gp=torch.autograd.grad(p,E)[0]
            rows[kind].append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
    report={}
    for kind,values in rows.items():
        a=np.array(values);denom=np.sqrt(np.mean(a[:,1]**2))
        report[kind]={'lambda':.1*np.sqrt(np.mean(a[:,0]**2))/denom if denom else 0.,'norms':values}
    report['absolute']['lambda']=json.loads((core.PRIOR/'coefficient.json').read_text())['lambda']
    ex.dump('calibration.json',report);return report


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    for j in range(370,740):
        population=data['old_only'] if spec.get('route')=='old_only' else np.arange(ex.SIZE)
        selected=rng.choice(population,32,replace=False)
        loss=ex.base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p=penalty(E,data,selected,spec['kind']);gp=torch.autograd.grad(p,E)[0]
        if 'geometry' in spec:gp=geometry(gp,gl,E,spec['geometry'])
        bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
        alpha=min(cal[spec['kind']]['lambda'],.1*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True)
        if spec.get('postadam'):
            before=E.detach().clone();E.grad=gl;opt.step()
            dn=float((E[ex.N:].detach()-before[ex.N:]).double().norm())
            beta=spec['rho']*dn/pn if pn else 0.
            with torch.no_grad():E.add_(gp,alpha=-beta)
        else:
            beta=None;E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha,
                        'gradient_ratio':alpha*pn/bn if bn else 0.,'postadam_beta':beta})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'frozen_rows_equal':True,'steps':370}
    ex.base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(core.PRIOR/'cache/prefix10.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();data=enrich(core.prepare_data(initial,old,new),new,len(initial['R']))
    cal=calibration(initial,prefix,new,tape,data);costs={'prepare_seconds':time.monotonic()-tick,'training':{}}
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,cal)
        ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs)
    ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
