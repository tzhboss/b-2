"""Fixed pressure thresholds for online global protection onset."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,'/data/lc/untzh/blend_confirm80_20261014')
import experiment80 as core
base=core.base;ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/blend_confirm80_20261014')
ARMS={'fixed5':None,'risk005':.005,'risk010':.01}

def load_data(seed):
    folder=SOURCE/f'seed{seed}';z=base.load(folder/'cache/cohort.npz')
    q=torch.as_tensor(z['q'],device='cuda');initial=torch.load(core.SRC/f'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s{seed}/initial.pt',map_location='cpu',weights_only=True)
    d=dict(q=q,candidates=torch.as_tensor(z['candidates'],device='cuda'),logZ=torch.as_tensor(z['logZ'],device='cuda'),legal_newcount=torch.as_tensor(z['legal_newcount'],device='cuda'))
    d['gold']=(q*initial['E'][z['queries'][:,2]].cuda().double()).sum(-1)
    d['selections']=np.stack([np.random.default_rng(seed+1000+j).choice(core.SIZE,32,replace=False) for j in range(555)])
    return initial,d,base.load(folder/'cache/learning.npz'),base.load(folder/'cache/teacher-train.npz')['old_train']

@torch.no_grad()
def pressure(E,d):
    torch.cuda.synchronize();tick=time.monotonic();count=0
    for lo in range(0,core.SIZE,256):
        scores=(d['q'][lo:lo+256,None]*E[d['candidates'][lo:lo+256]].double()).sum(-1)
        count+=int((scores>d['gold'][lo:lo+256,None]).any(-1).sum())
    torch.cuda.synchronize();return count/core.SIZE,time.monotonic()-tick

def prefix5(initial,new,t):
    tick=time.monotonic();E,R,opt=core.newmodel(initial)
    for j in range(185):
        loss=base.learning(E,R,new,t,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    assert torch.isfinite(E).all() and torch.equal(E[:core.N].cpu(),initial['E'][:core.N])
    p=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()))
    torch.save(p,core.ROOT/'prefix5.pt');return p,time.monotonic()-tick

def calibration(E,R,new,t,d,start):
    tick=time.monotonic();rows={s:[] for s in ['absolute','mass']}
    for k,j in enumerate(range(start,min(start+8,740))):
        gl=torch.autograd.grad(base.learning(E,R,new,t,j),E)[0]
        for s in rows:
            gp=torch.autograd.grad(core.penalty(E,d,d['selections'][k],s),E)[0]
            rows[s].append([float(gl[core.N:].double().norm()),float(gp[core.N:].double().norm())])
    c={}
    for s,v in rows.items():
        a=np.array(v);den=np.sqrt(np.mean(a[:,1]**2));assert den>0
        c[s]=.1*np.sqrt(np.mean(a[:,0]**2))/den
    return c,time.monotonic()-tick

def train(name,threshold,seed,initial,prefix,new,t,d,known):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix)
    onset=None;cal=None;detect_seconds=0.;cal_seconds=0.;observations=[];steps=[]
    for j in range(185,740):
        ep=int(t['epochs'][j])-1
        if j==185 or t['epochs'][j]!=t['epochs'][j-1]:
            if onset is None:
                value,seconds=pressure(E,d);detect_seconds+=seconds
                observations.append(dict(epoch=ep,pressure_fraction=value))
                if threshold is None or value>=threshold:
                    onset=ep
                    # Before first intervention all arms follow exactly the host.
                    # Same onset implies identical algorithm; reuse fixed controls.
                    if onset in known:
                        return dict(seconds=time.monotonic()-tick,detector_seconds=detect_seconds,onset=onset,observations=observations,
                            protected_steps=(20-onset)*37,reused_arm=known[onset],source='same first onset, same calibration and protection sequence')
                    cal,cal_seconds=calibration(E,R,new,t,d,j)
                    core.progress('triggered',arm=name,epoch=ep,pressure=value)
        loss=base.learning(E,R,new,t,j);gl=torch.autograd.grad(loss,E)[0];g=torch.zeros_like(gl)
        if onset is not None:
            bn=float(gl[core.N:].double().norm());chosen=d['selections'][j-onset*37]
            for s in ['absolute','mass']:
                p=core.penalty(E,d,chosen,s);gp=torch.autograd.grad(p,E)[0];pn=float(gp[core.N:].double().norm())
                alpha=min(cal[s],.1*bn/pn) if pn else 0.;g+=.5*alpha*gp
            assert float(g[core.N:].double().norm())<=.1*bn*(1+1e-5)
        opt.zero_grad(set_to_none=True);E.grad=gl+g;opt.step()
        assert torch.isfinite(loss) and torch.isfinite(E).all()
        steps.append(dict(step=j,epoch=ep+1,protected=onset is not None,loss=float(loss)))
        if j==739 or t['epochs'][j+1]!=t['epochs'][j]:core.progress('train',arm=name,epoch=ep+1)
    assert torch.equal(E[:core.N].cpu(),initial['E'][:core.N]) and torch.equal(R.cpu(),initial['R'])
    out=core.ROOT/'models'/name;out.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt')
    base.write(out/'steps.json',steps)
    result=dict(seconds=time.monotonic()-tick,detector_seconds=detect_seconds,calibration_seconds=cal_seconds,onset=onset,
        observations=observations,protected_steps=0 if onset is None else (20-onset)*37,frozen_equal=True)
    base.write(out/'training.json',result)
    if onset is not None:known[onset]=name
    return result

def run(seed):
    tick=time.monotonic();core.ROOT=ROOT/f'seed{seed}';core.ROOT.mkdir();base.ROOT=core.ROOT
    initial,d,t,A=load_data(seed);trdata=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=trdata['old'],trdata['new'];tr=core.truthmap(old,new)
    p,ps=prefix5(initial,new,t);costs=dict(prefix_seconds=ps,training={},preparation_source=str(SOURCE/f'seed{seed}'))
    known={10:'fixed10'}
    for arm,threshold in ARMS.items():costs['training'][arm]=train(arm,threshold,seed,initial,p,new,t,d,known);core.dump('costs-running.json',costs)
    # Read VALID only once every arm is committed or explicitly aliased.
    ev=core.ROOT/'evaluation';ev.mkdir();prior=__import__('json').loads((SOURCE/f'seed{seed}'/'evaluation/metrics.json').read_text())
    valid=base.load(core.SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    av=base.load(SOURCE/f'seed{seed}'/'evaluation/teacher.npz')['old_valid']
    report={'none':prior['none'],'fixed10':prior['blend']};es=time.monotonic()
    for arm in ARMS:
        c=costs['training'][arm]
        if 'reused_arm' in c:
            report[arm]=copy.deepcopy(report[c['reused_arm']]);report[arm]['reused_from']=c['reused_arm'];continue
        if c['onset'] is None:
            report[arm]=copy.deepcopy(prior['none']);report[arm]['reused_from']='none';continue
        m=torch.load(core.ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda()
        qs=dict(old_train=old,old_valid=valid['old'],new_valid=valid['new'])
        states={key:base.ranks(E,R,q,tr,arm=arm,split=key) for key,q in qs.items()};np.savez(ev/f'{arm}.npz',**states)
        report[arm]={key:base.metrics(states[key],q) for key,q in qs.items()}
        for key,teacher_a in [('old_train',A),('old_valid',av)]:report[arm][key]['drops']=core.drops(states[key],teacher_a)
        report[arm]['directional_ratios']=[report[arm]['new_valid'][x]['filtered_train_truths']['mrr']/prior['none']['new_valid'][x]['filtered_train_truths']['mrr'] for x in core.DIRECTIONS]
        core.progress('evaluated',arm=arm,onset=c['onset'],new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    costs['evaluation_seconds']=time.monotonic()-es;costs['total_seconds']=time.monotonic()-tick
    core.dump('evaluation/metrics.json',report);core.dump('costs.json',costs);core.progress('complete',seed=seed,seconds=costs['total_seconds'])
    return report,costs

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reports={};costs={};tick=time.monotonic()
    for seed in [29,43]:reports[str(seed)],costs[str(seed)]=run(seed)
    core.ROOT=ROOT;core.dump('summary.json',dict(metrics=reports,costs=costs,total_seconds=time.monotonic()-tick));core.progress('complete')

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
