import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/data/lc/untzh/query_space92_20261014')
import experiment92 as prior
action, route, core, base = prior.action, prior.route, prior.core, prior.base
ROOT=Path(__file__).resolve().parent
prior.ROOT=action.ROOT=route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=ROOT
N=prior.N
ARMS=['joint_null','joint_normmatched']
route.ARMS=ARMS


def load_space():
    tick=time.monotonic()
    space=torch.load('/data/lc/untzh/query_space92_20261014/cache/query-space.pt',map_location='cpu',weights_only=True)
    for g in space['groups']:
        g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
    return space,time.monotonic()-tick


def train(arm,initial,prefix,new,tape,data,space):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix)
    records=[];proposal_seconds=0.;projection_seconds=0.
    for j in range(220,440):
        chosen=data['selections'][j-220]
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,_=action.penalty(E,data,chosen,'absolute_full');gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        before=E.detach().clone()
        torch.cuda.synchronize();st=time.monotonic()
        host_delta=route.ex.proposal(E,opt,gl)
        proposal_seconds+=time.monotonic()-st
        # Keep the same joint-gradient moments as the ordinary joint arm.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        joint=E.detach().clone();host=before+host_delta
        raw=(joint-host)[N:]
        torch.cuda.synchronize();st=time.monotonic()
        projected=prior.project(raw,space)
        projection_seconds+=time.monotonic()-st
        rn,pnstep=float(raw.double().norm()),float(projected.norm())
        assert pnstep<=rn*(1+1e-8)+1e-12
        correction=projected if arm=='joint_null' else raw.double()*(pnstep/rn if rn else 0.)
        with torch.no_grad():
            E.copy_(host);E[N:].add_(correction.to(E.dtype))
        actual=(E.detach()[N:]-host[N:]).double()
        changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=int(tape['epochs'][j]),raw_protection_norm=rn,
                            projected_norm=pnstep,actual_protection_norm=float(actual.norm()),
                            host_step_norm=float(host_delta[N:].double().norm()),
                            cross_train_max_score_change=float(changes.abs().max()),
                            cross_train_mean_abs_score_change=float(changes.abs().mean())))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:
            core.progress('train',arm=arm,epoch=int(tape['epochs'][j]),max_score_change=records[-1]['cross_train_max_score_change'])
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/arm;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records)
    result=dict(seconds=time.monotonic()-tick,steps=220,proposal_seconds_included=proposal_seconds,
                projection_seconds_included=projection_seconds,
                mean_raw_protection_norm=float(np.mean([x['raw_protection_norm'] for x in records])),
                mean_actual_protection_norm=float(np.mean([x['actual_protection_norm'] for x in records])),
                max_cross_train_score_change=max(x['cross_train_max_score_change'] for x in records),frozen_equal=True)
    base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare()
    es=action.extra_data(truths,data);space,ss=load_space()
    costs=dict(preparation_seconds=ps+es+ss,space_load_seconds=ss,
               reused_space='/data/lc/untzh/query_space92_20261014/cache/query-space.pt',training={})
    for arm in ARMS:
        costs['training'][arm]=train(arm,initial,prefix,new,tape,data,space)
        core.dump('costs-running.json',costs)
    metrics,es=route.evaluate(truths)
    saved=json.loads(Path('/data/lc/untzh/query_space92_20261014/summary.json').read_text())['metrics']
    for arm in ['post_raw','post_null','post_normmatched','joint_fullabsolute']:
        metrics[arm]=copy.deepcopy(saved[arm])
    costs.update(evaluation_seconds=es,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
