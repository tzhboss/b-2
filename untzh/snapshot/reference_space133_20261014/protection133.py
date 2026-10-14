import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import experiment131 as worker
base,cx=worker.base,worker.cx
_spaces={}

def references(parent,initial,n,total,truths,steps,out):
    tick=time.monotonic()
    path=Path('/data/lc/untzh/reference_action132_20261014')/str(parent)/'cache/reference.npz'
    z=base.load(path)
    assert z['legal'].shape==(len(z['ids']),total-n) and len(z['selections'])==steps
    data={k:torch.as_tensor(z[k],device='cuda') for k in ['q','gold','oldrank','legal']}
    data['qnorm']=data['q'].norm(dim=1).clamp_min(1e-30)
    data['selections']=z['selections']
    report=dict(seconds=time.monotonic()-tick,reference_queries=len(z['ids']),steps=steps,reused=str(path),
                extra_old_valid_calibration_supervision=True,historical_preparation_not_free=True)
    out.mkdir(parents=True,exist_ok=True)
    base.write(out/'reference.json',report)
    return data,report

def space_for(parent):
    if parent in _spaces:
        return _spaces[parent]
    path=Path('/data/lc/untzh/anchor_confirm95_20261014/cache/query-space.pt') if parent==17 else Path('/data/lc/untzh/complex_transfer118_20261014/cache/query-space.pt')
    s=torch.load(path,map_location='cpu',weights_only=True)
    for g in s['groups']:
        g['indices']=g['indices'].cuda()
        g['basis']=g['basis'].cuda()
    s['rows']=s['rows'].cuda()
    s['vectors']=s['vectors'].cuda()
    _spaces[parent]=s
    return s

def ordinary_path(parent,init):
    if init=='constrained':
        return Path('/data/lc/untzh/reference_train131_20261014')/str(parent)/'historical_reference/constrained/models/none/epoch-20.pt'
    return Path('/data/lc/untzh/train_init121_20261014/seed17/models/zero/epoch-20.pt') if parent==17 else Path('/data/lc/untzh/wn_init122_20261014/ComplEx/models/zero/epoch-20.pt')

def train(parent,arm,initial,prefix,n,new,tape,data,out):
    tick=time.monotonic()
    E,R,opt=worker.model(initial,n,prefix)
    se,sr,so=worker.model(initial,n,prefix)
    space=space_for(parent)
    learning=cx.learning if parent=='WN_ComplEx' else base.learning
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    rows=[]
    for j in range(start,len(tape['epochs'])):
        chosen=data['selections'][j-start]
        gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        scores=data['q'][chosen]@E[n:].double().T
        legal=data['legal'][chosen]
        p=(F.softplus(scores).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
        gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[n:].double().norm()),float(gp[n:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True)
        E.grad=gl+alpha*gp
        opt.step()
        so.zero_grad(set_to_none=True)
        learning(se,sr,new,tape,j).backward()
        so.step()
        ref=se.detach()[n:]
        raw=(E.detach()[n:]-ref).double()
        projected=cx.source.space_module.project(raw,space)
        raw_norm,pnorm=float(raw.norm()),float(projected.norm())
        assert pnorm<=raw_norm*(1+1e-8)+1e-12
        accepted=projected if arm=='reference_softplus_anchor' else raw*(pnorm/raw_norm if raw_norm else 0.)
        with torch.no_grad():
            E[n:].copy_(ref+accepted.to(E.dtype))
        actual=(E.detach()[n:]-ref).double()
        changes=(space['vectors']*actual[space['rows']]).sum(1)
        assert bool(torch.isfinite(E).all())
        rows.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,raw_norm=raw_norm,projected_norm=pnorm,
                         actual_norm=float(actual.norm()),cross_max=float(changes.abs().max()),protection_norm=pn))
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    ordinary=torch.load(ordinary_path(parent,out.name),map_location='cpu',weights_only=True)
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    directory=out/'models'/arm
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',rows)
    report=dict(seconds=time.monotonic()-tick,steps=len(rows),nonzero_steps=sum(r['protection_norm']>0 for r in rows),
                mean_ratio=float(np.mean([r['ratio'] for r in rows])),shadow_compute_included=True,shadow_equal=True,
                max_cross_score_difference=max(r['cross_max'] for r in rows),final_norm=rows[-1]['actual_norm'],frozen_equal=True)
    base.write(directory/'training.json',report)
    return report
