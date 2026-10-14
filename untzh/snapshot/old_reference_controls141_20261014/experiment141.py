import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/entity_action137_20261014')
import experiment137 as action
sys.path.insert(0,'/data/lc/untzh/risk_link136_20261014')
import experiment136 as risk
worker,base,core,cx=action.worker,action.base,action.core,action.cx
ROOT=Path(__file__).resolve().parent
PILOT=Path('/data/lc/untzh/old_reference_pilot140_20261014')
SRC=Path('/data/lc/untzh/wn_transfer112_20261014')
N,TOTAL,D=24567,28660,200
ARMS=['none','fixed_anchor','fixed_replay']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def prepare(initial,old,new,tape,truths):
    tick=action.sync()
    cx.ROOT=ROOT
    teacher,teacher_cost=cx.teacher_cache(initial,old,truths,'old_train')
    eligible=np.flatnonzero(teacher['teacher'][:,1]<=10)
    size=min(16384,len(eligible))
    assert size>=32
    ids=np.random.default_rng(17).choice(eligible,size,replace=False)
    q=cx.tail_query(initial['E'][old[ids,0]].double(),initial['R'][old[ids,1]].double())
    legal=np.ones((size,TOTAL-N),bool)
    for i,(h,r,g) in enumerate(old[ids]):
        legal[i,[v-N for v in truths[int(h),int(r)] if v>=N]]=False
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    steps=len(tape['epochs'])-start
    selections=np.stack([np.random.default_rng(17+1000+j).choice(size,32,replace=False) for j in range(steps)])
    np.savez(ROOT/'cache/cohort.npz',ids=ids,queries=old[ids],q=q.numpy(),legal=legal,selections=selections)
    anchor=dict(q=q.cuda(),legal=torch.as_tensor(legal,device='cuda'),selections=selections)
    space,space_cost=cx.build_space(initial,new)
    replayz=base.load('/data/lc/untzh/complex_transfer118_20261014/cache/full-replay.npz')
    replayfacts=old[replayz['positions'][start*32:]]
    assert len(replayfacts)==steps*32
    replayz['qt']=cx.tail_query(initial['E'][replayfacts[:,0]].double(),initial['R'][replayfacts[:,1]].double()).numpy().reshape(steps,32,D)
    replayz['qh']=cx.head_query(initial['E'][replayfacts[:,2]].double(),initial['R'][replayfacts[:,1]].double()).numpy().reshape(steps,32,D)
    np.savez(ROOT/'cache/full-replay.npz',**replayz)
    replay={k:torch.as_tensor(replayz[k],device='cuda') for k in ['qt','qh','mt','mh','dt','dh']}
    split=base.load(PILOT/'cache/split.npz')
    report=dict(seconds=action.sync()-tick,teacher=teacher_cost,space=space_cost,cohort_size=size,
                cohort_internal_reference=int((~split['fit_mask'][ids]).sum()),
                replay_internal_reference_draws=int((~split['fit_mask'][replayz['positions'][start*32:]]).sum()),
                historical_pilot_parent_not_free=True,tape_no_resampling=True)
    return teacher,anchor,replay,space,report

def benefit_report(teacher,phases,endpoints,gold,norm,mask):
    oldrank=teacher[:,1]
    ids=np.flatnonzero(mask&(oldrank<=10))
    prefix=phases['prefix10'][ids,1]
    ordinary=endpoints['none'][ids,1]
    scores=dict(prefix_rank=prefix,oldrank=oldrank[ids],weak_gold=-gold[ids]/np.maximum(norm[ids],1e-30),
                prefix_excess=prefix-10)
    result=dict(eligible=len(ids),ordinary_drops=int((ordinary>10).sum()),arms={})
    for arm in ['fixed_anchor','fixed_replay']:
        after=endpoints[arm][ids,1]
        rescue=(ordinary>10)&(after<=10)
        harm=(ordinary<=10)&(after>10)
        result['arms'][arm]=dict(extra_rescues=int(rescue.sum()),new_harm=int(harm.sum()),
            improved_rr=int((after<ordinary).sum()),worse_rr=int((after>ordinary).sum()),
            mean_rr_gain=float((1/after-1/ordinary).mean()) if len(ids) else None,
            rescue_prediction={name:risk.performance(rescue,value) for name,value in scores.items()},
            association_not_necessity=True)
    return result

def evaluate(initial,prefix,ordinary,old,truths,traincache,fit_mask):
    tick=action.sync()
    # The first official VALID access occurs only after all endpoints are saved.
    saved=base.load(SRC/'evaluation/queries.npz')
    queries=dict(old_train=old,old_valid=saved['old_valid'],new_valid=saved['new_valid'])
    cx.ROOT=ROOT
    validcache,teacher_cost=cx.teacher_cache(initial,queries['old_valid'],truths,'old_valid')
    caches=dict(old_train=traincache,old_valid=validcache)
    teacher={s:z['teacher'] for s,z in caches.items()}
    split=base.load('/data/lc/untzh/admission_action128_20261014/cache/WN_ComplEx-split.npz')
    metric,drift,states,costs={},{},{},{}
    for arm in ARMS:
        status('evaluate',arm=arm)
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        assert torch.equal(m['E'][:N],initial['E'][:N].double()) and torch.equal(m['R'],initial['R'].double())
        E,R=m['E'].cuda(),m['R'].cuda()
        armstates={}
        armcost={}
        for s in ['old_train','old_valid']:armstates[s],armcost[s]=cx.fast_ranks(E,R,queries[s],caches[s],truths)
        t=action.sync()
        armstates['new_valid']=cx.ranks(E,R,queries['new_valid'],truths)
        armcost['new_valid_seconds']=action.sync()-t
        np.savez(ROOT/'evaluation'/f'{arm}.npz',**armstates)
        stats=worker.metric(armstates,queries,teacher,split)
        for name,mask in [('old_fit',fit_mask),('old_internal_reference',~fit_mask)]:
            stats[name]=base.metrics(armstates['old_train'][mask],old[mask])
            stats[name]['drops']=core.drops(armstates['old_train'][mask],teacher['old_train'][mask])
        metric[arm]=stats
        drift[arm]={s:action.drift(armstates[s],teacher[s]) for s in ['old_train','old_valid']}
        states[arm]=armstates
        costs[arm]=armcost
        print('EVALUATED141',arm,stats['new_valid']['all']['filtered_train_truths']['mrr'],
              stats['old_valid']['drops']['filtered_train_truths']['drops'],stats['old_internal_reference']['drops']['filtered_train_truths']['drops'],flush=True)
    for arm in ARMS:
        metric[arm]['directional_ratios_to_none']=[metric[arm]['new_valid'][d]['filtered_train_truths']['mrr']/metric['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
    phases={}
    phasecost={}
    benefits={}
    for s in ['old_train','old_valid']:
        phases[s]={}
        phasecost[s]={}
        for label,m in [('expanded',initial),('prefix10',prefix)]:
            phases[s][label],phasecost[s][label]=cx.fast_ranks(m['E'].cuda().double(),m['R'].cuda().double(),queries[s],caches[s],truths)
        np.savez(ROOT/'evaluation'/f'{s}-phases.npz',teacher=teacher[s],**phases[s])
        q=cx.tail_query(initial['E'][queries[s][:,0]].double(),initial['R'][queries[s][:,1]].double())
        norm=q.norm(dim=1).numpy()
        if s=='old_train':masks=dict(fit=fit_mask,internal_reference=~fit_mask)
        else:masks=dict(development_valid=np.ones(len(queries[s]),bool))
        for name,mask in masks.items():
            benefits[name]=benefit_report(teacher[s],phases[s],{arm:states[arm][s] for arm in ARMS},caches[s]['gold'],norm,mask)
    return dict(metrics=metric,drift=drift,benefit_diagnostic=benefits,
                costs=dict(seconds=action.sync()-tick,valid_teacher=teacher_cost,endpoints=costs,phases=phasecost))

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    base.N=core.N=N
    base.TOTAL=core.TOTAL=TOTAL
    tick=time.monotonic()
    for name in ['cache','models','evaluation']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'models/fixed_anchor/epoch-20.pt').exists(), 'do not silently restart existing controls'
    status('prepare')
    initial=torch.load(PILOT/'cache/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(PILOT/'models/prefix10.pt',map_location='cpu',weights_only=True)
    ordinary=torch.load(PILOT/'models/epoch20.pt',map_location='cpu',weights_only=True)
    z=base.load(SRC/'cache/train.npz')
    tape=base.load(SRC/'cache/learning.npz')
    truths=core.truthmap(z['old'],z['new'])
    traincache,anchor,replay,space,prep=prepare(initial,z['old'],z['new'],tape,truths)
    training={}
    for arm,code in [('none',0),('fixed_anchor',1),('fixed_replay',2)]:
        status('train',arm=arm)
        actions=np.full(TOTAL-N,code,np.int8)
        training[arm]=action.train('WN_ComplEx',arm,initial,prefix,ordinary,N,z['new'],tape,anchor,replay,space,actions,ROOT)
        print('TRAIN141',arm,training[arm],flush=True)
    none=torch.load(ROOT/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
    assert torch.equal(none['E'],ordinary['E']) and torch.equal(none['R'],ordinary['R'])
    base.write(ROOT/'committed-models.json',dict(training_trajectories=3,none_equals_pilot=True,all_endpoints_before_current_valid=True))
    fit=base.load(PILOT/'cache/split.npz')['fit_mask']
    result=evaluate(initial,prefix,ordinary,z['old'],truths,traincache,fit)
    result.update(preparation=prep,training=training,total_seconds=time.monotonic()-tick,
                  new_training_configs=3,new_parent_trainings=0,none_not_independent_repeat=True,
                  no_test=True,all_valid_development=True,adaptive_method_not_tested=True)
    base.write(ROOT/'summary.json',result)
    status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
