import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/joint_space93_20261014')
import experiment93 as previous
prior,action,route,core,base=previous.prior,previous.action,previous.route,previous.core,previous.base
ROOT=Path(__file__).resolve().parent
previous.ROOT=prior.ROOT=action.ROOT=route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=ROOT
N=prior.N
ARMS=['anchor_null','anchor_normmatched']
route.ARMS=ARMS


def anchors(initial,prefix,new,tape):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix);states=[]
    for j in range(220,440):
        loss=base.learning(E,R,new,tape,j)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        states.append(E.detach()[N:].cpu().clone())
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:core.progress('prepare_anchor',epoch=int(tape['epochs'][j]))
    ordinary=torch.load(route.SOURCE/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
    assert torch.equal(E.detach().cpu().double(),ordinary['E']) and torch.equal(R.cpu().double(),ordinary['R'])
    states=torch.stack(states);(ROOT/'cache').mkdir()
    torch.save(states,ROOT/'cache/ordinary-anchors.pt')
    return states,dict(seconds=time.monotonic()-tick,bytes=states.numel()*states.element_size(),ordinary_endpoint_equal=True)


def train(arm,initial,prefix,new,tape,data,space,anchors):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix);records=[];projection_seconds=0.
    for j in range(220,440):
        chosen=data['selections'][j-220]
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,_=action.penalty(E,data,chosen,'absolute_full');gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchors[j-220].cuda()
        displacement=(E.detach()[N:]-reference).double()
        torch.cuda.synchronize();start=time.monotonic();projected=prior.project(displacement,space)
        projection_seconds+=time.monotonic()-start
        raw_norm,pnorm=float(displacement.norm()),float(projected.norm())
        assert pnorm<=raw_norm*(1+1e-8)+1e-12
        accepted=projected if arm=='anchor_null' else displacement*(pnorm/raw_norm if raw_norm else 0.)
        with torch.no_grad():E[N:].copy_(reference+accepted.to(E.dtype))
        actual=(E.detach()[N:]-reference).double()
        changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=int(tape['epochs'][j]),raw_anchor_displacement_norm=raw_norm,
                            accepted_anchor_displacement_norm=float(actual.norm()),projected_norm=pnorm,
                            max_cross_train_score_difference=float(changes.abs().max()),
                            mean_abs_cross_train_score_difference=float(changes.abs().mean())))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:
            core.progress('train',arm=arm,epoch=int(tape['epochs'][j]),max_score_difference=records[-1]['max_cross_train_score_difference'])
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/arm;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records)
    result=dict(seconds=time.monotonic()-tick,steps=220,projection_seconds_included=projection_seconds,
                final_anchor_displacement_norm=records[-1]['accepted_anchor_displacement_norm'],
                max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records),frozen_equal=True)
    base.write(out/'training.json',result);return result


def main():
    torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare()
    es=action.extra_data(truths,data);space,ss=previous.load_space();states,ac=anchors(initial,prefix,new,tape)
    costs=dict(preparation_seconds=ps+es+ss,anchors=ac,reused_space='/data/lc/untzh/query_space92_20261014/cache/query-space.pt',training={})
    for arm in ARMS:
        costs['training'][arm]=train(arm,initial,prefix,new,tape,data,space,states)
        core.dump('costs-running.json',costs)
    metrics,es=route.evaluate(truths)
    saved=previous.json.loads(Path('/data/lc/untzh/joint_space93_20261014/summary.json').read_text())['metrics']
    for arm in ['joint_null','joint_normmatched','joint_fullabsolute']:metrics[arm]=saved[arm]
    costs.update(evaluation_seconds=es,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
