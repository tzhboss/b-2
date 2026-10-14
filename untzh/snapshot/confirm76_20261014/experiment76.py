"""Fixed candidate onset robustness and exact matched-control completion."""
import collections
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

SOURCE=Path('/data/lc/untzh/stratified75_20261014')
sys.path.insert(0,str(SOURCE))
import experiment75 as prev
core=prev.core;ex=prev.ex
ROOT=Path(__file__).resolve().parent
prev.ROOT=ROOT;core.ROOT=ROOT;ex.ROOT=ROOT;ex.base.ROOT=ROOT
ARMS={f'onset{epoch}_{method}':(epoch,power,weight,rho)
      for epoch in [0,5] for method,power,weight,rho in [
       ('candidate',1.,1.1,.15),('matched_uniform',0.,1.1,.15),
       ('simple_uniform',0.,1.,.1),('learning_only',0.,1.1,0.)]}


def prefix_states(initial,new,tape):
    E,R,opt=ex.init_model(initial)
    zero={'E':initial['E'],'R':initial['R'],'optimizer':copy.deepcopy(opt.state_dict())}
    for j in range(185):
        loss=ex.base.learning(E,R,new,tape,j);g=torch.autograd.grad(loss,E)[0]
        opt.zero_grad(set_to_none=True);E.grad=g;opt.step()
    five={'E':E.detach().cpu(),'R':R.cpu(),'optimizer':copy.deepcopy(opt.state_dict())}
    torch.save(five,ROOT/'prefix5.pt')
    return {0:zero,5:five}


def calibrate(initial,prefix,new,tape,data,start,spec):
    _,power,weight,rho=spec
    if not rho:return {'lambda':0.}
    E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);rows=[]
    for j in range(start,start+8):
        selected=prev.sampling(data,power,rng)
        gl=torch.autograd.grad(prev.learning(E,R,new,tape,j,weight),E)[0]
        gp=torch.autograd.grad(prev.prev.penalty(E,data,selected),E)[0]
        rows.append([float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())])
    a=np.array(rows);return {'lambda':rho*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2)),'norms':rows}


def train(name,spec,initial,prefix,new,tape,data,cal):
    started=time.monotonic();epoch0,power,weight,rho=spec;start=epoch0*37
    E,R,opt=core.model(initial,prefix);rng=np.random.default_rng(17);records=[]
    for j in range(start,740):
        selected=prev.sampling(data,power,rng)
        loss=prev.learning(E,R,new,tape,j,weight);gl=torch.autograd.grad(loss,E)[0]
        p=E.new_zeros(());alpha=0.;gp=torch.zeros_like(E)
        if rho:
            p=prev.prev.penalty(E,data,selected);gp=torch.autograd.grad(p,E)[0]
            bn,pn=float(gl[ex.N:].double().norm()),float(gp[ex.N:].double().norm())
            alpha=min(cal['lambda'],rho*bn/pn) if pn else 0.
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(E).all():raise FloatingPointError(name)
        records.append({'step':j,'epoch':int(tape['epochs'][j]),'loss':float(loss),'penalty':float(p),'alpha':alpha})
        if j==739 or tape['epochs'][j+1]!=tape['epochs'][j]:ex.progress('train',arm=name,epoch=int(tape['epochs'][j]),seconds=time.monotonic()-started)
    assert torch.equal(E[:ex.N].detach().cpu(),initial['E'][:ex.N]) and torch.equal(R.cpu(),initial['R'])
    out=ROOT/'models'/name;out.mkdir(parents=True)
    torch.save({'E':E.detach().cpu().double(),'R':R.cpu().double()},out/'epoch-20.pt')
    ex.base.write(out/'steps.json',records)
    result={'seconds':time.monotonic()-started,'spec':spec,'steps':740-start,'old_frozen_equal':True}
    ex.base.write(out/'training.json',result);return result


@torch.no_grad()
def supplement_controls(old,new):
    start=time.monotonic();truth=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new]):truth[int(h),int(r)].add(int(t))
    A=ex.base.load(ex.SRC/'replay_strength53_20261013/artifacts/evaluation/A.npz')
    sources=[(SOURCE,'relation_w1.1_rho0.15'),(SOURCE,'uniform_w1.1_rho0.15'),
             (SOURCE,'relation_w1.0_rho0.1'),(Path('/data/lc/untzh/coverage74_20261014'),'global_uniform32')]
    report={}
    for source,name in sources:
        item=json.loads((source/'evaluation/metrics.json').read_text())[name]
        if 'old_train' not in item:
            m=torch.load(source/'models'/name/'epoch-20.pt',map_location='cpu',weights_only=True)
            a=ex.base.ranks(m['E'].cuda(),m['R'].cuda(),old,truth,arm=name,split='old_train')
            np.savez(ROOT/'evaluation'/f'reused-{name}-train.npz',old_train=a)
            item['old_train']=ex.base.metrics(a,old);item['old_train']['drops']=core.dropstats(a,A['old_train'])
        item['source']=str(source)
        report[name]=item
        ex.progress('matched_control_complete',arm=name,drops=item['old_train']['drops']['filtered_train_truths']['drops'])
    ex.dump('evaluation/onset10-controls.json',report)
    return {'seconds':time.monotonic()-start,'sources':sources.__len__()}


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();ex.progress('starting')
    initial=torch.load(ex.SRC/'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',map_location='cpu',weights_only=True)
    d=ex.base.load(ex.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz');old,new=d['old'],d['new']
    tape=ex.base.load(ex.SRC/'matched_protection52_20261013/artifacts/tape/learning.npz')
    tick=time.monotonic();cached=ex.base.load(SOURCE/'shared-cache.npz')
    data={'q':torch.as_tensor(cached['q'],device='cuda'),'candidates':torch.as_tensor(cached['candidates'],device='cuda'),
          'eligible':cached['eligible'],'queries':cached['queries'],'relation_prob':cached['relation_prob']}
    prefixes=prefix_states(initial,new,tape)
    costs={'prepare_seconds':time.monotonic()-tick,'cache_source':str(SOURCE/'shared-cache.npz'),'training':{}}
    cal={name:calibrate(initial,prefixes[spec[0]],new,tape,data,spec[0]*37,spec) for name,spec in ARMS.items()}
    ex.dump('calibration.json',cal)
    for name,spec in ARMS.items():
        costs['training'][name]=train(name,spec,initial,prefixes[spec[0]],new,tape,data,cal[name]);ex.dump('costs-running.json',costs)
    _,costs['evaluation']=core.evaluate(old,new,ARMS)
    costs['supplement']=supplement_controls(old,new)
    costs['total_seconds']=time.monotonic()-started;ex.dump('costs.json',costs);ex.progress('complete',seconds=costs['total_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:ex.progress('failed',error=repr(exc));raise
