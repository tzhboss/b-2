import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/anchor_route97_20261014')
import experiment97 as previous
prior,action,route,core,base=previous.prior,previous.action,previous.route,previous.core,previous.base
ROOT=Path(__file__).resolve().parent
previous.ROOT=previous.previous.ROOT=prior.ROOT=action.ROOT=route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=ROOT
N,SIZE=prior.N,core.SIZE
ARMS=['query_tree','query_random_pool','query_uniform']
route.ARMS=ARMS


def train(arm,initial,prefix,new,tape,data,model,space,anchors,representatives,inverse):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix);rng=np.random.default_rng(1010029)
    count=len(representatives);records=[];decisions=[];detection_seconds=0.
    for j in range(220,440):
        ep=int(tape['epochs'][j])-1
        if j==220 or tape['epochs'][j-1]!=tape['epochs'][j]:
            torch.cuda.synchronize();start=time.monotonic()
            if arm=='query_tree':
                X=route.features(E,data,ep)
                if j==220:
                    frame=base.load(route.SOURCE/'cache/frame-10.npz')['X'];cols=list(range(15))+[20]
                    assert np.max(np.abs(X[:,cols]-frame[:,cols]))<=1e-6
                answer_scores=model.predict_proba(X)[:,1];scores=np.zeros(count)
                np.maximum.at(scores,inverse,answer_scores)
            elif arm=='query_random_pool':scores=np.random.default_rng(1010000+ep).random(count)
            else:scores=None
            pool=None if scores is None else np.argsort(-scores,kind='stable')[:int(np.ceil(.05*count))]
            torch.cuda.synchronize();seconds=time.monotonic()-start;detection_seconds+=seconds
            decisions.append(dict(epoch=ep,seconds=seconds,pool=None if pool is None else pool.tolist()))
        if arm=='query_uniform':group_ids=rng.choice(count,32,replace=False)
        else:
            first=rng.choice(pool,26,replace=False);available=np.ones(count,bool);available[first]=False
            group_ids=np.concatenate([first,rng.choice(np.flatnonzero(available),6,replace=False)])
        chosen=representatives[group_ids]
        assert len(np.unique(inverse[chosen]))==32
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,_=action.penalty(E,data,chosen,'absolute_full');gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchors[j-220].cuda();raw=(E.detach()[N:]-reference).double();projected=prior.project(raw,space)
        assert float(projected.norm())<=float(raw.norm())*(1+1e-8)+1e-12
        with torch.no_grad():E[N:].copy_(reference+projected.to(E.dtype))
        actual=(E.detach()[N:]-reference).double();changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=ep+1,selected=chosen.tolist(),query_groups=group_ids.tolist(),
                            accepted_anchor_displacement_norm=float(actual.norm()),max_cross_train_score_difference=float(changes.abs().max())))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:core.progress('train',arm=arm,epoch=ep+1)
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/arm;out.mkdir(parents=True);torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records);base.write(out/'decisions.json',decisions)
    exposures=np.bincount(np.concatenate([x['query_groups'] for x in records]),minlength=count);weights=exposures/exposures.sum()
    report=dict(seconds=time.monotonic()-tick,detection_seconds_included=detection_seconds,steps=220,frozen_equal=True,
                unique_queries=int((exposures>0).sum()),effective_query_exposure=float(1/(weights*weights).sum()),
                max_query_exposure=int(exposures.max()),final_anchor_displacement_norm=records[-1]['accepted_anchor_displacement_norm'],
                max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records))
    base.write(out/'training.json',report);return report


def main():
    torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare();action.extra_data(truths,data)
    space,ss=previous.previous.previous.load_space()
    anchors=torch.load('/data/lc/untzh/anchor_space94_20261014/cache/ordinary-anchors.pt',map_location='cpu',weights_only=True)
    c=base.load(route.SOURCE/'cache/cohort.npz');groups,representatives,inverse=np.unique(c['queries'][:,:2],axis=0,return_index=True,return_inverse=True)
    assert np.array_equal(c['q'],c['q'][representatives[inverse]])
    (ROOT/'cache').mkdir();np.savez(ROOT/'cache/query-groups.npz',groups=groups,representatives=representatives,inverse=inverse)
    costs=dict(preparation_seconds=time.monotonic()-tick,query_groups=len(groups),cohort_answers=SIZE,historical_assets_not_free=True,training={})
    for arm in ARMS:
        costs['training'][arm]=train(arm,initial,prefix,new,tape,data,model,space,anchors,representatives,inverse)
        core.dump('costs-running.json',costs)
    metrics,es=route.evaluate(truths)
    historical=json.loads(Path('/data/lc/untzh/risk_action100_20261014/summary.json').read_text())['metrics']
    for name,metric in historical.items():
        if name not in metrics:metrics[name]=metric
    costs.update(evaluation_seconds=es,total_seconds=time.monotonic()-tick)
    core.dump('summary.json',dict(metrics=metrics,costs=costs));core.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
