import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, '/data/lc/untzh/reference_init130_20261014')
import experiment130 as initialization
cx, base, core = initialization.cx, initialization.base, initialization.core
sys.path.insert(0, '/data/lc/untzh/fast_old_eval107_20261014')
import experiment107 as fastmodule
ROOT = Path(__file__).resolve().parent
CONDITIONS = ['train_only', 'historical_reference']
DIRECTIONS = ['constrained', 'normmatched_random', 'reverse']

def status(stage, **kw):
    base.write(ROOT/'status.json', dict(stage=stage, pid=os.getpid(), **kw))

def model(initial, n, prefix=None):
    E = torch.nn.Parameter(initial['E'].cuda().clone())
    R = initial['R'].cuda().clone()
    E.register_hook(lambda g: torch.cat([torch.zeros_like(g[:n]), g[n:]]))
    opt = torch.optim.Adam([E], lr=.007)
    if prefix:
        E.data.copy_(prefix['E'].cuda())
        opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    return E, R, opt

def train_none(initial, n, complex_model, new, tape, out):
    tick = time.monotonic()
    E,R,opt = model(initial,n)
    learning = cx.learning if complex_model else base.learning
    directory = out/'models/none'
    directory.mkdir(parents=True, exist_ok=True)
    start = int(np.flatnonzero(tape['epochs']==11)[0])
    prefix = None
    for j in range(len(tape['epochs'])):
        opt.zero_grad(set_to_none=True)
        learning(E,R,new,tape,j).backward()
        opt.step()
        assert bool(torch.isfinite(E).all())
        if j == start-1:
            prefix = dict(E=E.detach().cpu(), R=R.cpu(), optimizer=copy.deepcopy(opt.state_dict()))
            torch.save(prefix, out/'prefix10.pt')
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    ordinary=dict(E=E.detach().cpu().double(), R=R.cpu().double())
    torch.save(ordinary, directory/'epoch-20.pt')
    return prefix,ordinary,dict(seconds=time.monotonic()-tick,steps=len(tape['epochs']),frozen_equal=True)

def train_anchor(initial, prefix, ordinary, n, complex_model, new, tape, data, space, out):
    tick=time.monotonic()
    E,R,opt=model(initial,n,prefix)
    se,sr,so=model(initial,n,prefix)
    learning=cx.learning if complex_model else base.learning
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    records=[]
    for j in range(start,len(tape['epochs'])):
        k=j-start
        gl=torch.autograd.grad(learning(E,R,new,tape,j),E)[0]
        chosen=data['selections'][k]
        scores=data['q'][chosen]@E[n:].double().T
        legal=data['legal'][chosen]
        penalty=(F.softplus(scores).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean()
        gp=torch.autograd.grad(penalty,E)[0]
        bn,pn=float(gl[n:].double().norm()),float(gp[n:].double().norm())
        alpha=.1*bn/pn if pn else 0.
        opt.zero_grad(set_to_none=True)
        E.grad=gl+alpha*gp
        opt.step()
        so.zero_grad(set_to_none=True)
        learning(se,sr,new,tape,j).backward()
        so.step()
        ref=se.detach()[n:]
        delta=(E.detach()[n:]-ref).double()
        accepted=cx.source.space_module.project(delta,space)
        with torch.no_grad():
            E[n:].copy_(ref+accepted.to(E.dtype))
        actual=(E.detach()[n:]-ref).double()
        changes=(space['vectors']*actual[space['rows']]).sum(1)
        assert bool(torch.isfinite(E).all())
        records.append(dict(step=j,ratio=alpha*pn/bn if bn else 0.,cross_max=float(changes.abs().max()),norm=float(actual.norm())))
    assert torch.equal(E[:n].cpu(),initial['E'][:n]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    directory=out/'models/answer_anchor'
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',records)
    return dict(seconds=time.monotonic()-tick,steps=len(records),shadow_compute_included=True,shadow_equal=True,
                max_cross_score_difference=max(r['cross_max'] for r in records),mean_ratio=float(np.mean([r['ratio'] for r in records])),frozen_equal=True)

def training_assets(parent):
    fb=parent==17
    src=Path('/data/lc/untzh')/('increment87_20261014' if fb else 'complex_transfer118_20261014')
    n,total=(11258,13134) if fb else (24567,28660)
    train_src=src if fb else Path('/data/lc/untzh/wn_transfer112_20261014')
    z=base.load(train_src/'cache/train.npz')
    truths=core.truthmap(z['old'],np.concatenate([z['new'],z['extra']]) if fb else z['new'])
    tape=base.load(train_src/'cache/learning.npz')
    cohort=base.load(src/'cache/cohort.npz')
    legal=cohort.get('legal')
    if legal is None:
        legal=np.ones((len(cohort['queries']),total-n),bool)
        for i,(h,r,g) in enumerate(cohort['queries']):
            legal[i,[v-n for v in truths[int(h),int(r)] if v>=n]]=False
    data=dict(q=torch.as_tensor(cohort['q'],device='cuda'),legal=torch.as_tensor(legal,device='cuda'))
    data['selections']=base.load(src/'cache/protection.npz')['selections'] if fb else cohort['selections']
    spacepath=Path('/data/lc/untzh/anchor_confirm95_20261014/cache/query-space.pt') if fb else src/'cache/query-space.pt'
    space=torch.load(spacepath,map_location='cpu',weights_only=True)
    for g in space['groups']:
        g['indices']=g['indices'].cuda()
        g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda()
    space['vectors']=space['vectors'].cuda()
    return n,total,src,z['new'],tape,truths,data,space

def transitions(teacher,initial,prefix,endpoint,mask):
    d0=initial>10
    d1=prefix>10
    d2=endpoint>10
    return dict(eligible=int(mask.sum()),expanded_drops=int((d0&mask).sum()),prefix_drops=int((d1&mask).sum()),endpoint_drops=int((d2&mask).sum()),
                initial_to_prefix=dict(new=int((~d0&d1&mask).sum()),rescued=int((d0&~d1&mask).sum())),
                prefix_to_endpoint=dict(new=int((~d1&d2&mask).sum()),rescued=int((d1&~d2&mask).sum())))

def metric(states,queries,teacher,split):
    result={s:base.metrics(states[s],q) for s,q in queries.items()}
    for s in ['old_train','old_valid']:
        result[s]['drops']=core.drops(states[s],teacher[s])
    for name,mask in [('old_cal',split['calibration']),('old_eval',split['evaluation'])]:
        result[name]=base.metrics(states['old_valid'][mask],queries['old_valid'][mask])
        result[name]['drops']=core.drops(states['old_valid'][mask],teacher['old_valid'][mask])
    return result

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic()
    bundles=[]
    for parent in [17,'WN_ComplEx']:
        status('prepare',parent=str(parent))
        n,total,src,new,tape,truths,data,space=training_assets(parent)
        for condition in CONDITIONS:
            for direction in DIRECTIONS:
                status('train',parent=str(parent),condition=condition,direction=direction)
                out=ROOT/str(parent)/condition/direction
                out.mkdir(parents=True,exist_ok=True)
                initial=torch.load(initialization.ROOT/str(parent)/condition/f'{direction}-initial.pt',map_location='cpu',weights_only=True)
                prefix,ordinary,ordinary_cost=train_none(initial,n,parent=='WN_ComplEx',new,tape,out)
                anchor_cost=train_anchor(initial,prefix,ordinary,n,parent=='WN_ComplEx',new,tape,data,space,out)
                costs=dict(none=ordinary_cost,answer_anchor=anchor_cost,initialization_from_130=True)
                base.write(out/'training.json',costs)
                bundles.append((parent,n,total,src,truths,condition,direction,out,costs))
                print('TRAIN131',parent,condition,direction,costs,flush=True)
    base.write(ROOT/'committed-models.json',dict(training_configs=24,all_endpoints_saved_before_current_valid=True))
    reports={}
    for parent in [17,'WN_ComplEx']:
        n,total,src,_,_,truths,_,_=training_assets(parent)
        base.N=core.N=n
        base.TOTAL=core.TOTAL=total
        split=base.load(f'/data/lc/untzh/admission_action128_20261014/cache/{parent}-split.npz')
        if parent==17:
            fastmodule.ROOT=Path('/data/lc/untzh/fast_old_eval107_20261014')
            fast=fastmodule.FastOld(17,truths)
            queries=dict(fast.queries,new_valid=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['new_valid'])
            teacher=fast.teacher
            oldrank=lambda E,R,s:fast.ranks(E,R,s)
            newrank=lambda E,R:base.ranks(E,R,queries['new_valid'],truths)
        else:
            queries=base.load(src/'evaluation/queries.npz')
            cache={s:base.load(src/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']}
            teacher={s:cache[s]['teacher'] for s in cache}
            oldrank=lambda E,R,s:cx.fast_ranks(E,R,queries[s],cache[s],truths)
            newrank=lambda E,R:cx.ranks(E,R,queries['new_valid'],truths)
        pred=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{parent}-old_valid-half-prediction.npz')
        ids=pred['ids']
        group_masks=dict(calibration=split['calibration'][ids],held_group=split['evaluation'][ids])
        parent_report=dict(cases={},fixed_controls={})
        for b in [x for x in bundles if x[0]==parent]:
            _,_,_,_,_,condition,direction,out,costs=b
            status('evaluate',parent=str(parent),condition=condition,direction=direction)
            evaluation=out/'evaluation'
            evaluation.mkdir(exist_ok=True)
            pref=torch.load(out/'prefix10.pt',map_location='cpu',weights_only=True)
            prefix_rank,prefix_cost=oldrank(pref['E'].cuda().double(),pref['R'].cuda().double(),'old_valid')
            init=base.load(initialization.ROOT/str(parent)/'evaluation'/f'{condition}-{direction}.npz')['rank']
            metrics={}
            phases={}
            ec={}
            for arm in ['none','answer_anchor']:
                state=torch.load(out/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
                E,R=state['E'].cuda(),state['R'].cuda()
                ranks={}
                cost={}
                for s in ['old_train','old_valid']:
                    ranks[s],cost[s]=oldrank(E,R,s)
                st=time.monotonic()
                ranks['new_valid']=newrank(E,R)
                cost['new_valid']=dict(seconds=time.monotonic()-st)
                np.savez(evaluation/f'{arm}.npz',**ranks)
                metrics[arm]=metric(ranks,queries,teacher,split)
                ec[arm]=cost
                phases[arm]={group:transitions(teacher['old_valid'][ids,1],init,prefix_rank[ids,1],ranks['old_valid'][ids,1],mask) for group,mask in group_masks.items()}
                print('EVALUATED131',parent,condition,direction,arm,metrics[arm]['new_valid']['all']['filtered_train_truths']['mrr'],metrics[arm]['old_eval']['drops']['filtered_train_truths']['drops'],flush=True)
            for arm in metrics:
                metrics[arm]['directional_ratios_to_own_none']=[metrics[arm]['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            report=dict(metrics=metrics,phases=phases,costs=dict(training=costs,evaluation=ec,prefix_scoring=prefix_cost))
            parent_report['cases'][condition+'-'+direction]=report
            base.write(out/'summary.json',report)
        for mode in ['zero','random_half']:
            folder=Path('/data/lc/untzh/train_init121_20261014/seed17') if parent==17 else Path('/data/lc/untzh/wn_init122_20261014/ComplEx')
            protected=Path('/data/lc/untzh/smallinit_protect126_20261014/seed17')/mode if parent==17 else Path('/data/lc/untzh/wn_smallprotect127_20261014/ComplEx')/mode
            ordinary=base.load(folder/'evaluation'/f'{mode}.npz')
            anchored=base.load(protected/'evaluation'/('anchor_null.npz' if parent==17 else 'answer_anchor.npz'))
            parent_report['fixed_controls'][mode]=dict(none=metric(ordinary,queries,teacher,split),answer_anchor=metric(anchored,queries,teacher,split))
        reports[str(parent)]=parent_report
        base.write(ROOT/str(parent)/'summary.json',parent_report)
        base.write(ROOT/'summary-running.json',dict(parents=reports))
    base.write(ROOT/'summary.json',dict(parents=reports,total_seconds=time.monotonic()-tick,new_training_configs=24,all_valid_development=True))
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
