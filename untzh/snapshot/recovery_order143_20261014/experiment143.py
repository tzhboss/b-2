import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/proposal_choice142_20261014')
import experiment142 as previous
action,worker,base,core,cx=previous.action,previous.worker,previous.base,previous.core,previous.cx
ROOT=Path(__file__).resolve().parent
CONTROL,PILOT,SRC=previous.CONTROL,previous.PILOT,previous.SRC
N,TOTAL,D=previous.N,previous.TOTAL,previous.D
ARMS=['none','fixed_anchor','fixed_replay','random_order','recovery_first','recovery_last']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def assets(initial,old,truths):
    tick=action.sync()
    cohort=base.load(CONTROL/'cache/cohort.npz')
    anchor={k:torch.as_tensor(cohort[k],device='cuda') for k in ['q','legal']}
    anchor['selections']=cohort['selections']
    replay={k:torch.as_tensor(v,device='cuda') for k,v in base.load(CONTROL/'cache/full-replay.npz').items() if k!='positions'}
    space=torch.load(CONTROL/'cache/query-space.pt',map_location='cpu',weights_only=True)
    for g in space['groups']:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    space['rows']=space['rows'].cuda();space['vectors']=space['vectors'].cuda()
    selected=base.load(Path('/data/lc/untzh/proposal_choice142_20261014/cache/fixed-probes.npz'))['old_reference_ids']
    teacher=base.load(CONTROL/'cache/fast-old_train.npz')
    facts=old[selected]
    q=cx.tail_query(initial['E'][facts[:,0]].double(),initial['R'][facts[:,1]].double())
    legal=np.ones((len(selected),TOTAL-N),bool)
    for i,(h,r,g) in enumerate(facts):legal[i,[v-N for v in truths[int(h),int(r)] if v>=N]]=False
    ref=dict(q=q.cuda(),gold=torch.as_tensor(teacher['gold'][selected],device='cuda'),
             oldrank=torch.as_tensor(teacher['teacher'][selected,1],device='cuda'),legal=torch.as_tensor(legal,device='cuda'))
    np.savez(ROOT/'cache/reference.npz',ids=selected,q=q.numpy(),gold=teacher['gold'][selected],oldrank=teacher['teacher'][selected,1],legal=legal)
    report=dict(seconds=action.sync()-tick,reference_probe=len(selected),probe_142_reused=True,
                no_resampling=True,no_future_endpoint_labels=True,no_current_valid_input=True,historical_assets_not_free=True)
    return anchor,replay,space,ref,teacher,report

@torch.no_grad()
def score_reference(E,ref):
    count=((ref['q']@E[N:].double().T>ref['gold'][:,None])&ref['legal']).sum(1)
    rank=ref['oldrank']+count
    return dict(mrr=float((1/rank.double()).mean()),drops=int((rank>10).sum()))

def train(arm,initial,prefix,ordinary,new,tape,anchor,replay,space,ref):
    E,R,opt=worker.model(initial,N,prefix)
    ce,cr,co=worker.model(initial,N,prefix)
    se,sr,so=worker.model(initial,N,prefix)
    start=int(np.flatnonzero(tape['epochs']==11)[0])
    rng=np.random.default_rng(1430017)
    costs={k:0. for k in ['preview_gradients_and_adam','preview_ranking','learning_and_protection','ordinary_shadow','update_and_projection','recording']}
    epochs=[]
    steps=[]
    submitted=[]
    begin=action.sync()
    for epoch in np.unique(tape['epochs'][start:]):
        ids=np.flatnonzero(tape['epochs']==epoch)
        scores=[]
        tick=action.sync()
        common=copy.deepcopy(opt.state_dict())
        costs['preview_gradients_and_adam']+=action.sync()-tick
        for j in ids:
            tick=action.sync()
            gl=torch.autograd.grad(cx.learning(E,R,new,tape,int(j)),E)[0]
            ce.data.copy_(E.detach())
            co.load_state_dict(copy.deepcopy(common))
            co.zero_grad(set_to_none=True)
            ce.grad=gl
            co.step()
            costs['preview_gradients_and_adam']+=action.sync()-tick
            tick=action.sync()
            scores.append(score_reference(ce,ref))
            costs['preview_ranking']+=action.sync()-tick
        if arm=='recovery_first':order=sorted(range(len(ids)),key=lambda i:(-scores[i]['mrr'],int(ids[i])))
        elif arm=='recovery_last':order=sorted(range(len(ids)),key=lambda i:(scores[i]['mrr'],int(ids[i])))
        elif arm=='random_order':order=rng.permutation(len(ids)).tolist()
        else:order=list(range(len(ids)))
        batch_ids=[int(ids[i]) for i in order]
        assert sorted(batch_ids)==ids.tolist()
        epochs.append(dict(epoch=int(epoch),original_ids=ids.tolist(),submitted_ids=batch_ids,predictions=scores,
                           distinct_predicted_mrr=len(set(s['mrr'] for s in scores))))
        for j in batch_ids:
            k=len(submitted)
            tick=action.sync()
            gl=torch.autograd.grad(cx.learning(E,R,new,tape,j),E)[0]
            chosen=anchor['selections'][k]
            ss=anchor['q'][chosen]@E[N:].double().T
            legal=anchor['legal'][chosen]
            ga=torch.autograd.grad((F.softplus(ss).masked_fill(~legal,0).sum(1)/legal.sum(1)).mean(),E)[0]
            tail=F.softplus(replay['qt'][k]@E[N:].double().T).masked_fill(~replay['mt'][k],0).sum(1)/replay['dt'][k]
            head=F.softplus(replay['qh'][k]@E[N:].double().T).masked_fill(~replay['mh'][k],0).sum(1)/replay['dh'][k]
            gr=torch.autograd.grad(.5*(tail+head).mean(),E)[0]
            gp=ga if arm=='fixed_anchor' else gr if arm=='fixed_replay' else torch.zeros_like(gl)
            bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
            alpha=.1*bn/pn if pn else 0.
            costs['learning_and_protection']+=action.sync()-tick
            tick=action.sync()
            so.zero_grad(set_to_none=True)
            cx.learning(se,sr,new,tape,j).backward()
            so.step()
            reference=se.detach()[N:]
            costs['ordinary_shadow']+=action.sync()-tick
            tick=action.sync()
            opt.zero_grad(set_to_none=True)
            E.grad=gl+alpha*gp
            opt.step()
            projected=cx.source.space_module.project((E.detach()[N:]-reference).double(),space)
            if arm=='fixed_anchor':
                with torch.no_grad():E[N:].copy_(reference+projected.to(E.dtype))
            costs['update_and_projection']+=action.sync()-tick
            tick=action.sync()
            assert torch.isfinite(E).all()
            submitted.append(j)
            steps.append(dict(step=k,original_batch=j,epoch=int(epoch),ratio=.1 if pn else 0.))
            costs['recording']+=action.sync()-tick
    assert sorted(submitted)==list(range(start,len(tape['epochs'])))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    if arm in ['none','fixed_anchor','fixed_replay']:
        assert torch.equal(se.detach().cpu().double(),ordinary['E']) and torch.equal(sr.cpu().double(),ordinary['R'])
    else:assert torch.equal(E.detach(),se.detach())
    out=ROOT/'models'/arm
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'epochs.json',epochs)
    base.write(out/'steps.json',steps)
    report=dict(seconds=action.sync()-begin,phase_costs=costs,committed_learning_steps=len(submitted),preview_steps=len(submitted),
                every_original_batch_once_each_epoch=True,draw_negative_pairs_unchanged=True,
                changed_batch_positions=sum(j!=start+k for k,j in enumerate(submitted)),
                mean_distinct_preview_scores=float(np.mean([e['distinct_predicted_mrr'] for e in epochs])),
                protection_step_fraction=1. if arm.startswith('fixed_') else 0.,
                mean_gradient_ratio=.1 if arm.startswith('fixed_') else 0.,
                equal_preview_gradient_shadow_projection_envelope=True,frozen_equal=True,
                shadow_follows_same_submitted_sequence=True)
    base.write(out/'training.json',report)
    return report

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    base.N=core.N=N
    base.TOTAL=core.TOTAL=TOTAL
    begin=time.monotonic()
    for folder in ['cache','models','evaluation']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'models/recovery_first/epoch-20.pt').exists(), 'existing endpoints: do not silently restart'
    status('prepare')
    initial=torch.load(PILOT/'cache/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load(PILOT/'models/prefix10.pt',map_location='cpu',weights_only=True)
    ordinary=torch.load(PILOT/'models/epoch20.pt',map_location='cpu',weights_only=True)
    z=base.load(SRC/'cache/train.npz')
    tape=base.load(SRC/'cache/learning.npz')
    truths=core.truthmap(z['old'],z['new'])
    anchor,replay,space,ref,teacher,prep=assets(initial,z['old'],truths)
    training={}
    for arm in ARMS:
        status('train',arm=arm)
        training[arm]=train(arm,initial,prefix,ordinary,z['new'],tape,anchor,replay,space,ref)
        print('TRAIN143',arm,training[arm],flush=True)
    base.write(ROOT/'committed-models.json',dict(training_trajectories=6,all_endpoints_before_current_valid=True,
        same_draws_negatives_steps=True,only_within_epoch_order_changed=True))
    previous.ROOT=ROOT
    previous.ARMS=ARMS
    fit=base.load(PILOT/'cache/split.npz')['fit_mask']
    result=previous.evaluate(initial,z['old'],truths,teacher,fit)
    result.update(preparation=prep,training=training,total_seconds=time.monotonic()-begin,new_training_configs=6,
                  new_parent_trainings=0,no_test=True,all_valid_development=True,equal_compute_envelope=True,
                  go_no_go_requires_joint_fixed_random_advantage=True)
    base.write(ROOT/'summary.json',result)
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
