"""Epoch-level action comparison on fixed TRAIN probes, with true Adam forks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import collections
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/action_guard82_20261014')
import experiment82 as prev
core=prev.core;base=prev.base;ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;base.ROOT=ROOT
ARMS=['local','cumulative','local_mean','local_shuffle','cumulative_mean','cumulative_shuffle']

def read(path):return json.loads(Path(path).read_text())

def prepare(initial,new,t,old):
    tick=time.monotonic();d,probes,pc=prev.prepare(initial,new,t)
    tr=core.truthmap(old,new)
    rows=np.random.default_rng(840017).choice(core.SIZE,1024,replace=False)
    z=base.load('/data/lc/untzh/risk58_20261013/cache/cohort.npz')
    oldmask=np.ones((1024,core.TOTAL-core.N),bool)
    for i,(h,r,g) in enumerate(z['queries'][rows]):
        ids=[c-core.N for c in tr[int(h),int(r)] if c>=core.N]
        oldmask[i,ids]=False
    newq=torch.cat([x['p'] for x in probes]);newmask=np.ones((len(newq),core.TOTAL),bool)
    for i,(h,r,g) in enumerate(newq.cpu().numpy()):
        ids=[c for c in tr[int(h),int(r)] if c!=int(g)];newmask[i,ids]=False
    pd=dict(old_q=d['q'][rows],old_gold=d['gold'][rows],old_rank=torch.as_tensor(z['oldrank'][rows],device='cuda'),
        old_legal=torch.as_tensor(oldmask,device='cuda'),new_q=newq,new_legal=torch.as_tensor(newmask,device='cuda'))
    np.savez(ROOT/'probe-cache.npz',old_cohort_rows=rows,old_legal=oldmask,new_queries=newq.cpu().numpy(),new_legal=newmask)
    return d,pd,tr,dict(seconds=time.monotonic()-tick,base_preparation=pc,old_train_answers=1024,new_train_answers_per_direction=256)

@torch.no_grad()
def score(snapshot,initial,pd):
    torch.cuda.synchronize();tick=time.monotonic();E=snapshot['E'].cuda().double();R=initial['R'].cuda().double()
    s=pd['old_q']@E[core.N:].T
    added=((s>pd['old_gold'][:,None])&pd['old_legal']).sum(-1)
    oldrank=pd['old_rank']+added
    old_mrr=float((1./oldrank.double()).mean());old_drops=int((oldrank>10).sum())
    q=pd['new_q'];s=(E[q[:,0]]*R[q[:,1]])@E.T
    gold=s[torch.arange(len(q),device='cuda'),q[:,2]].clone()
    nr=1+((s>gold[:,None])&pd['new_legal']).sum(-1)
    new_mrr=[float((1./nr[i*256:(i+1)*256].double()).mean()) for i in range(3)]
    assert np.isfinite(new_mrr).all() and old_mrr>0
    torch.cuda.synchronize()
    return dict(old_mrr=old_mrr,old_drops=old_drops,new_mrr=new_mrr,seconds=time.monotonic()-tick)

def branch(initial,snapshot,new,t,d,coefficient,epoch,multiplier):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,snapshot);ratios=[]
    for j in range(epoch*37,(epoch+1)*37):
        gl=torch.autograd.grad(base.learning(E,R,new,t,j),E)[0];g=torch.zeros_like(gl)
        if multiplier:
            chosen=d['selections'][j-370];p=core.penalty(E,d,chosen,'absolute')
            gp=torch.autograd.grad(p,E)[0];bn=float(gl[core.N:].double().norm());pn=float(gp[core.N:].double().norm())
            alpha=min(coefficient,.1*bn/pn) if pn else 0.;g=multiplier*alpha*gp
            ratio=float(g[core.N:].double().norm())/bn if bn else 0.;assert ratio<=.1*(1+1e-5);ratios.append(ratio)
        opt.zero_grad(set_to_none=True);E.grad=gl+g;opt.step();assert torch.isfinite(E).all()
    assert torch.equal(E[:core.N].cpu(),initial['E'][:core.N]) and torch.equal(R.cpu(),initial['R'])
    snap=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()))
    return snap,dict(seconds=time.monotonic()-tick,steps=37,max_gradient_ratio=max(ratios) if ratios else 0.)

def shadow(initial,prefix,new,t,d,pd,c):
    tick=time.monotonic();snap=prefix;records=[]
    for ep in range(10,20):
        snap,tc=branch(initial,snap,new,t,d,c,ep,0.);sc=score(snap,initial,pd)
        records.append(dict(epoch=ep+1,training=tc,probe=sc));core.progress('shadow',epoch=ep+1)
    torch.save(snap,ROOT/'shadow-terminal.pt');core.dump('shadow.json',dict(records=records,seconds=time.monotonic()-tick))
    return records,time.monotonic()-tick

def save(name,snapshot,result,records,initial):
    out=ROOT/'models'/name;out.mkdir(parents=True)
    assert torch.equal(snapshot['E'][:core.N],initial['E'][:core.N]) and torch.equal(snapshot['R'],initial['R'])
    torch.save(dict(E=snapshot['E'].double(),R=snapshot['R'].double()),out/'epoch-20.pt')
    base.write(out/'epochs.json',records);base.write(out/'training.json',result)

def adaptive(name,initial,prefix,new,t,d,pd,c,shadow_records):
    tick=time.monotonic();snap=prefix;records=[];seq=[];branches_seconds=0.;probes_seconds=0.
    for ep in range(10,20):
        options=[]
        for gamma in [0.,.5,1.]:
            candidate,tc=branch(initial,snap,new,t,d,c,ep,gamma);sc=score(candidate,initial,pd)
            options.append(dict(multiplier=gamma,snapshot=candidate,training=tc,probe=sc))
            branches_seconds+=tc['seconds'];probes_seconds+=sc['seconds']
        host=options[0]['probe'];ref=host['new_mrr'] if name=='local' else shadow_records[ep-10]['probe']['new_mrr']
        budget=.998 if name=='local' else .98
        qualified=[]
        for o in options:
            sc=o['probe'];r=np.array(sc['new_mrr'])/np.array(ref)
            o['direction_ratios']=r.tolist()
            o['budget_feasible']=bool((r>=budget).all())
            o['old_nonworse']=sc['old_mrr']>=host['old_mrr']-1e-14 and sc['old_drops']<=host['old_drops']
            if o['budget_feasible'] and o['old_nonworse']:qualified.append(o)
        chosen=max(qualified,key=lambda o:(o['probe']['old_mrr'],-o['probe']['old_drops'],-o['multiplier'])) if qualified else options[0]
        snap=chosen['snapshot'];seq.append(chosen['multiplier'])
        row=dict(epoch=ep+1,chosen=chosen['multiplier'],qualified_count=len(qualified),
            no_feasible_fallback=len(qualified)==0,reference=ref,options=[{k:v for k,v in o.items() if k!='snapshot'} for o in options])
        records.append(row);core.progress('decision',arm=name,epoch=ep+1,chosen=chosen['multiplier'],qualified=len(qualified),ratios=chosen['direction_ratios'])
    result=dict(seconds=time.monotonic()-tick,branch_training_seconds_included=branches_seconds,
        probe_seconds_included=probes_seconds,branches=30,selected_sequence=seq,mean_multiplier=float(np.mean(seq)),
        no_feasible_epochs=sum(x['no_feasible_fallback'] for x in records),frozen_equal=True)
    save(name,snap,result,records,initial);return result

def scheduled(name,sequence,initial,prefix,new,t,d,c):
    tick=time.monotonic();snap=prefix;records=[]
    for ep,gamma in zip(range(10,20),sequence):
        snap,tc=branch(initial,snap,new,t,d,c,ep,float(gamma));records.append(dict(epoch=ep+1,multiplier=float(gamma),training=tc))
        core.progress('train',arm=name,epoch=ep+1,multiplier=float(gamma))
    result=dict(seconds=time.monotonic()-tick,selected_sequence=list(map(float,sequence)),mean_multiplier=float(np.mean(sequence)),frozen_equal=True)
    save(name,snap,result,records,initial);return result

def evaluate(old,new,tr):
    tick=time.monotonic();valid=base.load(core.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    A=base.load(core.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    prior=read('/data/lc/untzh/risk58_20261013/evaluation/metrics.json');report=dict(none=prior['new_only'],absolute_fixed=prior['uniform'])
    known=dict(absolute_fixed=Path('/data/lc/untzh/risk58_20261013/models/uniform/epoch-20.pt'),
        none=core.SRC/'matched_protection52_20261013/artifacts/train/new_only/epoch-20.pt')
    # Actual none source is kept separate below if the historical path differs.
    out=ROOT/'evaluation';out.mkdir();aliases={}
    for arm in ARMS:
        path=ROOT/'models'/arm/'epoch-20.pt';m=torch.load(path,map_location='cpu',weights_only=True)
        alias=None
        for name,kpath in known.items():
            if not kpath.exists():continue
            k=torch.load(kpath,map_location='cpu',weights_only=True)
            if torch.equal(m['E'],k['E']) and torch.equal(m['R'],k['R']):alias=name;break
        if alias:
            report[arm]=copy.deepcopy(report[alias]);aliases[arm]=alias;continue
        E,R=m['E'].cuda(),m['R'].cuda();qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={k:base.ranks(E,R,q,tr,arm=arm,split=k) for k,q in qs.items()};np.savez(out/f'{arm}.npz',**states)
        report[arm]={k:base.metrics(states[k],q) for k,q in qs.items()}
        for k in ['old_train','old_valid']:report[arm][k]['drops']=core.drops(states[k],A[k])
        known[arm]=path;core.progress('evaluated',arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm,item in report.items():item['directional_ratios']=[item['new_valid'][x]['filtered_train_truths']['mrr']/report['none']['new_valid'][x]['filtered_train_truths']['mrr'] for x in core.DIRECTIONS]
    core.dump('evaluation/metrics.json',report);core.dump('evaluation/reused-equal-models.json',aliases)
    return report,time.monotonic()-tick

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting')
    initial=torch.load(core.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    prefix=torch.load('/data/lc/untzh/risk58_20261013/cache/prefix10.pt',map_location='cpu',weights_only=True)
    z=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=z['old'],z['new']
    t=base.load(core.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz');d,pd,tr,pc=prepare(initial,new,t,old)
    coefficient=read('/data/lc/untzh/normalized77_20261014/calibration.json')['cohort_blend0.5']['absolute']['lambda']
    sr,ss=shadow(initial,prefix,new,t,d,pd,coefficient);costs=dict(preparation=pc,shadow_seconds=ss,training={})
    for policy in ['local','cumulative']:
        costs['training'][policy]=adaptive(policy,initial,prefix,new,t,d,pd,coefficient,sr)
        seq=costs['training'][policy]['selected_sequence']
        costs['training'][policy+'_mean']=scheduled(policy+'_mean',[np.mean(seq)]*10,initial,prefix,new,t,d,coefficient)
        shuffled=np.random.default_rng(840017).permutation(seq)
        costs['training'][policy+'_shuffle']=scheduled(policy+'_shuffle',shuffled,initial,prefix,new,t,d,coefficient)
        core.dump('costs-running.json',costs)
    report,es=evaluate(old,new,tr);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
