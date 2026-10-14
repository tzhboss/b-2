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
sys.path.insert(0,'/data/lc/untzh/rank_confirm91_20261014')
import experiment91 as source17
prior,action,route,core,base=previous.prior,previous.action,previous.route,previous.core,previous.base
ROOT=Path(__file__).resolve().parent
N,SIZE=prior.N,core.SIZE


def configure(out,source):
    previous.ROOT=previous.previous.ROOT=source17.ROOT=prior.ROOT=action.ROOT=route.ROOT=route.prev.ROOT=core.ROOT=base.ROOT=out
    route.SOURCE=source


def status(stage,**extra):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**extra))


def load_space(path):
    space=torch.load(path,map_location='cpu',weights_only=True)
    for g in space['groups']:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
    return space


def train(name,spec,initial,prefix,new,tape,data,model,space,anchors):
    signal,policy=spec;tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix)
    rng=np.random.default_rng(890029);records=[];decisions=[];detection_seconds=0.
    for j in range(220,440):
        ep=int(tape['epochs'][j])-1
        if policy=='tree' and (j==220 or tape['epochs'][j-1]!=tape['epochs'][j]):
            torch.cuda.synchronize();start=time.monotonic();X=route.features(E,data,ep)
            if j==220:
                frame=base.load(route.SOURCE/'cache/frame-10.npz')['X'];cols=list(range(15))+[20]
                assert np.max(np.abs(X[:,cols]-frame[:,cols]))<=1e-6
            scores=model.predict_proba(X)[:,1];pool=np.argsort(-scores,kind='stable')[:int(np.ceil(.05*SIZE))]
            torch.cuda.synchronize();seconds=time.monotonic()-start;detection_seconds+=seconds
            decisions.append(dict(epoch=ep,seconds=seconds,pool=pool.tolist()))
        if policy=='uniform':chosen=data['selections'][j-220]
        else:
            first=rng.choice(pool,26,replace=False);available=np.ones(SIZE,bool);available[first]=False
            chosen=np.concatenate([first,rng.choice(np.flatnonzero(available),6,replace=False)])
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p,stats=action.penalty(E,data,chosen,signal);gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchors[j-220].cuda();raw=(E.detach()[N:]-reference).double();projected=prior.project(raw,space)
        assert float(projected.norm())<=float(raw.norm())*(1+1e-8)+1e-12
        with torch.no_grad():E[N:].copy_(reference+projected.to(E.dtype))
        actual=(E.detach()[N:]-reference).double();changes=(space['vectors']*actual[space['rows']]).sum(-1)
        assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=ep+1,selected=chosen.tolist(),penalty=float(p),gradient_ratio=alpha*pn/bn if bn else 0.,
                            accepted_anchor_displacement_norm=float(actual.norm()),max_cross_train_score_difference=float(changes.abs().max()),**stats))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:core.progress('train',arm=name,epoch=ep+1,active_pairs=stats['active_pairs'])
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    out=route.ROOT/'models'/name;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',records);base.write(out/'decisions.json',decisions)
    report=dict(seconds=time.monotonic()-tick,detection_seconds_included=detection_seconds,steps=220,frozen_equal=True,
                active_steps=sum(x['active_pairs']>0 for x in records),mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in records])),
                final_anchor_displacement_norm=records[-1]['accepted_anchor_displacement_norm'],
                max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records))
    base.write(out/'training.json',report);return report


def evaluate(seed,out,training_source,truths,names):
    tick=time.monotonic();directory=out/'evaluation';directory.mkdir()
    full=Path('/data/lc/untzh/full_valid102_20261014')
    queries=base.load(full/'queries.npz')
    source_queries=base.load(training_source/'evaluation/queries.npz')
    queries['old_train']=source_queries['old_train']
    teacher=dict(old_valid=base.load(full/f'seed{seed}/teacher.npz')['old_valid'],
                 old_train=base.load(training_source/'evaluation/teacher.npz')['old_train'])
    np.savez(directory/'queries.npz',**queries);np.savez(directory/'teacher.npz',**teacher)
    reference=json.loads((full/'summary.json').read_text())['seeds'][str(seed)]['metrics'];metrics={}
    for name in names:
        status('evaluate',seed=seed,arm=name)
        m=torch.load(out/'models'/name/'epoch-20.pt',map_location='cpu',weights_only=True)
        E,R=m['E'].cuda(),m['R'].cuda()
        states={split:base.ranks(E,R,q,truths,arm=f'{seed}_{name}',split=split) for split,q in queries.items()}
        np.savez(directory/f'{name}.npz',**states)
        metric={split:base.metrics(states[split],q) for split,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],teacher[split])
        metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
        metrics[name]=metric;base.write(directory/'metrics-running.json',metrics)
        print('EVALUATED103',dict(seed=seed,arm=name,new_mrr=metric['new_valid']['all']['filtered_train_truths']['mrr'],old_drops=metric['old_valid']['drops']['filtered_train_truths']['drops'],train_drops=metric['old_train']['drops']['filtered_train_truths']['drops']),flush=True)
    return metrics,time.monotonic()-tick,reference


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();status('starting');bundles=[]
    for seed in [29,17]:
        out=ROOT/f'seed{seed}';out.mkdir()
        training_source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        configure(out,training_source);torch.manual_seed(seed);status('prepare',seed=seed);prep=time.monotonic()
        if seed==29:
            initial,prefix,old,new,truths,tape,data,model,coefficient,ps=route.prepare();action.extra_data(truths,data)
            space=load_space('/data/lc/untzh/query_space92_20261014/cache/query-space.pt')
            anchor_source=Path('/data/lc/untzh/anchor_space94_20261014')
        else:
            initial,prefix,new,tape,truths,data,ps=source17.prepare();model=None
            space=load_space('/data/lc/untzh/anchor_confirm95_20261014/cache/query-space.pt')
            anchor_source=Path('/data/lc/untzh/anchor_confirm95_20261014')
        anchors=torch.load(anchor_source/'cache/ordinary-anchors.pt',map_location='cpu',weights_only=True)
        policies=['uniform','tree'] if seed==29 else ['uniform']
        specs={f'{signal}_{policy}':(signal,policy) for signal in ['violation_full','budget_full'] for policy in policies}
        costs=dict(preparation_seconds=time.monotonic()-prep,seed=seed,historical_assets_not_free=True,anchor_source=str(anchor_source),training={})
        for name,spec in specs.items():
            status('train',seed=seed,arm=name)
            costs['training'][name]=train(name,spec,initial,prefix,new,tape,data,model,space,anchors)
            core.dump('costs-running.json',costs)
        bundles.append((seed,out,training_source,truths,list(specs),costs))
    # No new training follows the first VALID read.
    summaries={}
    for seed,out,training_source,truths,names,costs in bundles:
        configure(out,training_source)
        metrics,seconds,reference=evaluate(seed,out,training_source,truths,names)
        costs['evaluation_seconds']=seconds
        summary=dict(metrics=metrics,controls_full_valid=reference,costs=costs)
        base.write(out/'summary.json',summary);summaries[str(seed)]=summary
    base.write(ROOT/'summary.json',dict(seeds=summaries,total_seconds=time.monotonic()-tick))
    status('complete',seconds=time.monotonic()-tick);print('COMPLETE103',time.monotonic()-tick,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
