import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/old_reference_controls141_20261014')
import experiment141 as controls
action,worker,base,core,cx=controls.action,controls.worker,controls.base,controls.core,controls.cx
ROOT=Path(__file__).resolve().parent
CONTROL=Path('/data/lc/untzh/old_reference_controls141_20261014')
PILOT=controls.PILOT
SRC=controls.SRC
N,TOTAL,D=controls.N,controls.TOTAL,controls.D
ARMS=['none','fixed_anchor','fixed_replay','pareto_choice','random_matched']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def assets(initial,old,new,tape,truths):
    tick=action.sync()
    cohort=base.load(CONTROL/'cache/cohort.npz')
    anchor={k:torch.as_tensor(cohort[k],device='cuda') for k in ['q','legal']}
    anchor['selections']=cohort['selections']
    replay={k:torch.as_tensor(v,device='cuda') for k,v in base.load(CONTROL/'cache/full-replay.npz').items() if k!='positions'}
    space=torch.load(CONTROL/'cache/query-space.pt',map_location='cpu',weights_only=True)
    for g in space['groups']:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
    traincache=base.load(CONTROL/'cache/fast-old_train.npz')
    split=base.load(PILOT/'cache/split.npz')
    qualified=np.flatnonzero((~split['fit_mask'])&(traincache['teacher'][:,1]<=10))
    selected=np.sort(np.random.default_rng(1420017).choice(qualified,min(256,len(qualified)),replace=False))
    assert len(selected)>=32
    facts=old[selected]
    q=cx.tail_query(initial['E'][facts[:,0]].double(),initial['R'][facts[:,1]].double())
    legal=np.ones((len(selected),TOTAL-N),bool)
    for i,(h,r,g) in enumerate(facts):legal[i,[v-N for v in truths[int(h),int(r)] if v>=N]]=False
    ref=dict(q=q.cuda(),gold=torch.as_tensor(traincache['gold'][selected],device='cuda'),
             oldrank=torch.as_tensor(traincache['teacher'][selected,1],device='cuda'),legal=torch.as_tensor(legal,device='cuda'))
    first=np.full(len(new),-1,np.int64)
    epoch1_end=int(tape['bounds'][int(np.flatnonzero(tape['epochs']==2)[0])])
    for i,p in enumerate(tape['pos'][:epoch1_end]):
        if first[p]<0:first[p]=i
    assert np.all(first>=0)
    masks=[(new[:,0]<N)&(new[:,2]>=N),(new[:,0]>=N)&(new[:,2]<N),(new[:,0]>=N)&(new[:,2]>=N)]
    rng=np.random.default_rng(1420017)
    probes=[]
    payload=dict(old_reference_ids=selected)
    for d,mask in enumerate(masks):
        candidates=np.flatnonzero(mask)
        chosen=np.sort(rng.choice(candidates,min(256,len(candidates)),replace=False))
        assert len(chosen)>0
        loc=first[chosen]
        probes.append(dict(facts=torch.as_tensor(new[chosen],device='cuda'),
                           nh=torch.as_tensor(tape['nh'][loc],device='cuda'),nt=torch.as_tensor(tape['nt'][loc],device='cuda')))
        payload[f'new_ids_{d}']=chosen
        payload[f'negative_positions_{d}']=loc
    np.savez(ROOT/'cache/fixed-probes.npz',**payload)
    report=dict(seconds=action.sync()-tick,old_reference_qualified=len(qualified),old_reference_probe=len(selected),
                new_probe_sizes=[len(v['facts']) for v in probes],seed=1420017,negative_tape_reused=True,
                no_endpoint_labels=True,no_current_valid_input=True,historical_140_141_cost_not_free=True)
    base.write(ROOT/'probes-frozen.json',report)
    return anchor,replay,space,ref,probes,traincache,split,report

@torch.no_grad()
def probe(E,R,ref,probes):
    count=((ref['q']@E[N:].double().T>ref['gold'][:,None])&ref['legal']).sum(1)
    rank=ref['oldrank']+count
    losses=[]
    for z in probes:
        h,r,t=z['facts'].unbind(1)
        positive=cx.score(E[h].double(),R[r].double(),E[t].double())
        negative=cx.score(E[z['nh']].double(),R[r[:,None]].double(),E[z['nt']].double())
        loss=F.softplus(-positive).mean()+F.softplus(negative).mean()+.0005*(E[h].double().square().mean()+R[r].double().square().mean())
        losses.append(float(loss))
    return dict(old_mrr=float((1/rank.double()).mean()),old_drops=int((rank>10).sum()),new_losses=losses)

def pick(scores):
    baseline=scores[0]
    admissible=[0]
    for code in [1,2]:
        s=scores[code]
        new_ok=all(v<=b+1e-12*(1+abs(b)) for v,b in zip(s['new_losses'],baseline['new_losses']))
        old_ok=s['old_drops']<=baseline['old_drops'] and s['old_mrr']>baseline['old_mrr']+1e-12
        if new_ok and old_ok:admissible.append(code)
    best=min(admissible,key=lambda k:(-scores[k]['old_mrr'],scores[k]['old_drops'],np.mean(scores[k]['new_losses']),k))
    return best,admissible

def train(arm,initial,prefix,ordinary,new,tape,anchor,replay,space,ref,probes,schedule=None):
    E,R,opt=worker.model(initial,N,prefix)
    candidates=[worker.model(initial,N,prefix) for _ in range(3)]
    se,sr,so=worker.model(initial,N,prefix)
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    costs={k:0. for k in ['gradients','ordinary_shadow','three_proposals_and_projection','probes','decision_commit','recording']}
    records=[]
    begin=action.sync()
    for j in range(start,len(tape['epochs'])):
        k=j-start
        tick=action.sync()
        gl=torch.autograd.grad(cx.learning(E,R,new,tape,j),E)[0]
        chosen=anchor['selections'][k]
        ss=anchor['q'][chosen]@E[N:].double().T
        pa=(F.softplus(ss).masked_fill(~anchor['legal'][chosen],0).sum(1)/anchor['legal'][chosen].sum(1)).mean()
        ga=torch.autograd.grad(pa,E)[0]
        tail=F.softplus(replay['qt'][k]@E[N:].double().T).masked_fill(~replay['mt'][k],0).sum(1)/replay['dt'][k]
        head=F.softplus(replay['qh'][k]@E[N:].double().T).masked_fill(~replay['mh'][k],0).sum(1)/replay['dh'][k]
        gr=torch.autograd.grad(.5*(tail+head).mean(),E)[0]
        bn=float(gl[N:].double().norm())
        gradients=[gl]
        for gp in [ga,gr]:
            pn=float(gp[N:].double().norm())
            gradients.append(gl+(.1*bn/pn if pn else 0.)*gp)
        costs['gradients']+=action.sync()-tick
        tick=action.sync()
        so.zero_grad(set_to_none=True)
        cx.learning(se,sr,new,tape,j).backward()
        so.step()
        reference=se.detach()[N:]
        costs['ordinary_shadow']+=action.sync()-tick
        tick=action.sync()
        common=copy.deepcopy(opt.state_dict())
        for code,(ce,cr,co) in enumerate(candidates):
            ce.data.copy_(E.detach())
            co.load_state_dict(copy.deepcopy(common))
            co.zero_grad(set_to_none=True)
            ce.grad=gradients[code]
            co.step()
        ce=candidates[1][0]
        projected=cx.source.space_module.project((ce.detach()[N:]-reference).double(),space)
        with torch.no_grad():ce[N:].copy_(reference+projected.to(ce.dtype))
        costs['three_proposals_and_projection']+=action.sync()-tick
        tick=action.sync()
        scores=[probe(ce,cr,ref,probes) for ce,cr,co in candidates]
        costs['probes']+=action.sync()-tick
        tick=action.sync()
        best,admissible=pick(scores)
        code={'none':0,'fixed_anchor':1,'fixed_replay':2}.get(arm,best)
        if schedule is not None:code=int(schedule[k])
        ce,cr,co=candidates[code]
        E.data.copy_(ce.detach())
        opt.load_state_dict(copy.deepcopy(co.state_dict()))
        costs['decision_commit']+=action.sync()-tick
        tick=action.sync()
        assert torch.isfinite(E).all()
        records.append(dict(step=j,action=code,pareto_action=best,admissible=admissible,probes=scores))
        costs['recording']+=action.sync()-tick
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    path=ROOT/'models'/arm
    path.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),path/'epoch-20.pt')
    base.write(path/'steps.json',records)
    sequence=np.array([r['action'] for r in records],np.int8)
    result=dict(seconds=action.sync()-begin,phase_costs=costs,committed_learning_steps=len(records),optimizer_proposals=3*len(records),
                actions={str(code):int((sequence==code).sum()) for code in [0,1,2]},
                protection_step_fraction=float((sequence>0).mean()),row_fraction_when_active=1.,
                mean_gradient_ratio=.1*float((sequence>0).mean()),same_three_proposals_and_probe_envelope=True,
                frozen_equal=True,ordinary_shadow_equal=True)
    base.write(path/'training.json',result)
    return sequence,result

def evaluate(initial,old,truths,traincache,fit_mask):
    tick=action.sync()
    queries=base.load(SRC/'evaluation/queries.npz')
    queries['old_train']=old
    cache=dict(old_train=traincache,old_valid=base.load(CONTROL/'cache/fast-old_valid.npz'))
    teacher={s:z['teacher'] for s,z in cache.items()}
    split=base.load('/data/lc/untzh/admission_action128_20261014/cache/WN_ComplEx-split.npz')
    metrics,drift,costs,reuse={},{},{},{}
    for arm in ARMS:
        status('evaluate',arm=arm)
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        assert torch.equal(m['E'][:N],initial['E'][:N].double()) and torch.equal(m['R'],initial['R'].double())
        equal=False
        if arm in ['none','fixed_anchor','fixed_replay']:
            previous=torch.load(CONTROL/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
            equal=torch.equal(m['E'],previous['E']) and torch.equal(m['R'],previous['R'])
        reuse[arm]=dict(parameter_equal_to_141=equal,not_independent_repeat=equal)
        armcost={}
        if equal:
            ranks=base.load(CONTROL/'evaluation'/f'{arm}.npz')
            armcost['existing_ranks_reused']=True
        else:
            E,R=m['E'].cuda(),m['R'].cuda()
            ranks={}
            for s in ['old_train','old_valid']:ranks[s],armcost[s]=cx.fast_ranks(E,R,queries[s],cache[s],truths)
            t=action.sync()
            ranks['new_valid']=cx.ranks(E,R,queries['new_valid'],truths)
            armcost['new_valid_seconds']=action.sync()-t
        np.savez(ROOT/'evaluation'/f'{arm}.npz',**ranks)
        metric=worker.metric(ranks,queries,teacher,split)
        for name,mask in [('old_fit',fit_mask),('old_internal_reference',~fit_mask)]:
            metric[name]=base.metrics(ranks['old_train'][mask],old[mask])
            metric[name]['drops']=core.drops(ranks['old_train'][mask],teacher['old_train'][mask])
        metrics[arm]=metric
        drift[arm]={s:action.drift(ranks[s],teacher[s]) for s in ['old_train','old_valid']}
        costs[arm]=armcost
        print('EVALUATED142',arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],
              metric['old_internal_reference']['drops']['filtered_train_truths']['drops'],flush=True)
    for arm in ARMS:
        metrics[arm]['directional_ratios_to_none']=[metrics[arm]['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
    return dict(metrics=metrics,drift=drift,reuse=reuse,evaluation_costs=costs,evaluation_seconds=action.sync()-tick)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    base.N=core.N=N
    base.TOTAL=core.TOTAL=TOTAL
    begin=time.monotonic()
    for folder in ['cache','models','evaluation']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'models/pareto_choice/epoch-20.pt').exists(), 'existing adaptive endpoint: do not restart silently'
    status('prepare')
    initial=torch.load(PILOT/'cache/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(PILOT/'models/prefix10.pt',map_location='cpu',weights_only=True)
    ordinary=torch.load(PILOT/'models/epoch20.pt',map_location='cpu',weights_only=True)
    z=base.load(SRC/'cache/train.npz')
    tape=base.load(SRC/'cache/learning.npz')
    truths=core.truthmap(z['old'],z['new'])
    anchor,replay,space,ref,probes,traincache,split,prep=assets(initial,z['old'],z['new'],tape,truths)
    training={}
    schedules={}
    for arm in ARMS:
        status('train',arm=arm)
        plan=None
        if arm=='random_matched':plan=np.random.default_rng(1420018).permutation(schedules['pareto_choice'])
        sequence,training[arm]=train(arm,initial,prefix,ordinary,z['new'],tape,anchor,replay,space,ref,probes,plan)
        schedules[arm]=sequence
        print('TRAIN142',arm,training[arm],flush=True)
    assert np.array_equal(np.bincount(schedules['pareto_choice'],minlength=3),np.bincount(schedules['random_matched'],minlength=3))
    np.savez(ROOT/'action-schedules.npz',**schedules)
    base.write(ROOT/'committed-models.json',dict(training_trajectories=5,all_endpoints_before_current_valid=True,
        random_action_counts_matched=True,random_posthoc_control=True,probe_input_train_only=True))
    result=evaluate(initial,z['old'],truths,traincache,split['fit_mask'])
    result.update(preparation=prep,training=training,total_seconds=time.monotonic()-begin,new_training_configs=5,
        new_parent_trainings=0,no_test=True,all_valid_development=True,equal_compute_envelope=True,
        go_no_go_requires_joint_fixed_random_advantage=True)
    base.write(ROOT/'summary.json',result)
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
