import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/epoch_guard84_20261014')
import experiment84 as prev
core=prev.core;base=prev.base;ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;prev.prev.ROOT=ROOT;core.ROOT=ROOT;base.ROOT=ROOT
ARMS=['matched_none','matched_absolute','local','cumulative','local_mean','local_shuffle','cumulative_mean','cumulative_shuffle']

def split(new,t):
    tick=time.monotonic();facts=new[new[:,1]%2==0];counts=np.bincount(facts[:,[0,2]].ravel(),minlength=core.TOTAL)
    hold=[];target=int(.2*len(facts));rng=np.random.default_rng(860017)
    for i in rng.permutation(len(facts)):
        h,r,g=map(int,facts[i]);need=2 if h==g else 1
        if counts[h]>need and counts[g]>need:
            hold.append(int(i));counts[h]-=1;counts[g]-=1
        if len(hold)==target:break
    heldfacts=facts[hold];keys=set()
    for h,r,g in heldfacts:keys.add((int(h),int(r),int(g)));keys.add((int(g),int(r)+1,int(h)))
    held=np.array([tuple(map(int,row)) in keys for row in new],bool)
    assert held.sum()==2*len(hold)
    retained_entities=np.unique(new[~held][:,[0,2]])
    assert set(np.unique(new[:,[0,2]]))<=set(retained_entities)
    pieces={k:[] for k in ['pos','nh','nt']};bounds=[0]
    for j in range(740):
        lo,hi=t['bounds'][j:j+2];keep=~held[t['pos'][lo:hi]]
        assert keep.any()
        for k in pieces:pieces[k].append(t[k][lo:hi][keep])
        bounds.append(bounds[-1]+int(keep.sum()))
    filtered={k:np.concatenate(v) for k,v in pieces.items()};filtered.update(bounds=np.array(bounds),epochs=t['epochs'])
    assert not held[filtered['pos']].any()
    np.savez(ROOT/'learning.npz',**filtered);np.savez(ROOT/'internal-split.npz',held_mask=held,held_facts=heldfacts,retained_entities=retained_entities)
    report=dict(seconds=time.monotonic()-tick,held_directed=int(held.sum()),retained_directed=int((~held).sum()),
        held_original=len(hold),target_original=target,retained_entity_count=len(retained_entities),learning_draws=len(filtered['pos']),
        original_draws=len(t['pos']),split_seed=860017,reverse_facts_excluded=True)
    core.dump('internal-split.json',report)
    return filtered,held,report

def prepare(initial,old,new,t,held):
    tick=time.monotonic();d,pd,tr,pc=prev.prepare(initial,new,t,old)
    rng=np.random.default_rng(860117)
    masks=[held&(new[:,0]<core.N)&(new[:,2]>=core.N),held&(new[:,0]>=core.N)&(new[:,2]<core.N),held&(new[:,0]>=core.N)&(new[:,2]>=core.N)]
    q=np.concatenate([new[rng.choice(np.flatnonzero(mask),256,replace=False)] for mask in masks])
    legal=np.ones((len(q),core.TOTAL),bool)
    for i,(h,r,g) in enumerate(q):legal[i,[c for c in tr[int(h),int(r)] if c!=int(g)]]=False
    pd['new_q']=torch.as_tensor(q,device='cuda');pd['new_legal']=torch.as_tensor(legal,device='cuda')
    np.savez(ROOT/'heldout-probes.npz',queries=q,legal=legal)
    return d,pd,tr,dict(seconds=time.monotonic()-tick,base=pc,probe_counts=[int(mask.sum()) for mask in masks])

def prefix(initial,new,t):
    tick=time.monotonic();E,R,opt=core.newmodel(initial)
    for j in range(370):
        loss=base.learning(E,R,new,t,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(E).all()
        if j==369 or t['epochs'][j+1]!=t['epochs'][j]:core.progress('prefix',epoch=int(t['epochs'][j]))
    assert torch.equal(E[:core.N].cpu(),initial['E'][:core.N])
    state=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()))
    torch.save(state,ROOT/'prefix10.pt');return state,time.monotonic()-tick

def evaluate(old,new,tr):
    tick=time.monotonic();valid=base.load(core.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    A=base.load(core.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    report={};known={};out=ROOT/'evaluation';out.mkdir();aliases={}
    for arm in ARMS:
        path=ROOT/'models'/arm/'epoch-20.pt';m=torch.load(path,map_location='cpu',weights_only=True);alias=None
        for name,kpath in known.items():
            k=torch.load(kpath,map_location='cpu',weights_only=True)
            if torch.equal(m['E'],k['E']) and torch.equal(m['R'],k['R']):alias=name;break
        if alias:report[arm]=copy.deepcopy(report[alias]);aliases[arm]=alias;continue
        E,R=m['E'].cuda(),m['R'].cuda();qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={key:base.ranks(E,R,q,tr,arm=arm,split=key) for key,q in qs.items()};np.savez(out/f'{arm}.npz',**states)
        report[arm]={key:base.metrics(states[key],q) for key,q in qs.items()}
        for key in ['old_train','old_valid']:report[arm][key]['drops']=core.drops(states[key],A[key])
        known[arm]=path;core.progress('evaluated',arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm,m in report.items():m['directional_ratios']=[m['new_valid'][x]['filtered_train_truths']['mrr']/report['matched_none']['new_valid'][x]['filtered_train_truths']['mrr'] for x in core.DIRECTIONS]
    core.dump('evaluation/metrics.json',report);core.dump('evaluation/reused-equal-models.json',aliases)
    return report,time.monotonic()-tick

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial=torch.load(core.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    z=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=z['old'],z['new']
    tape=base.load(core.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz');t,held,split_cost=split(new,tape)
    d,pd,tr,pc=prepare(initial,old,new,t,held);p,ps=prefix(initial,new,t)
    coeff,cs=core.calibrate(initial,p,new,t,d);c=coeff['absolute'];sr,ss=prev.shadow(initial,p,new,t,d,pd,c)
    costs=dict(split=split_cost,preparation=pc,prefix_seconds=ps,calibration_seconds=cs,shadow_seconds=ss,training={})
    for name,gamma in [('matched_none',0.),('matched_absolute',1.)]:costs['training'][name]=prev.scheduled(name,[gamma]*10,initial,p,new,t,d,c)
    for policy in ['local','cumulative']:
        costs['training'][policy]=prev.adaptive(policy,initial,p,new,t,d,pd,c,sr);seq=costs['training'][policy]['selected_sequence']
        costs['training'][policy+'_mean']=prev.scheduled(policy+'_mean',[np.mean(seq)]*10,initial,p,new,t,d,c)
        costs['training'][policy+'_shuffle']=prev.scheduled(policy+'_shuffle',np.random.default_rng(860017).permutation(seq),initial,p,new,t,d,c)
        core.dump('costs-running.json',costs)
    report,es=evaluate(old,new,tr);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
