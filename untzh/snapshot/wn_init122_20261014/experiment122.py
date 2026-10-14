import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import copy
import json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/complex_transfer118_20261014')
import experiment118 as cx
source,base,core=cx.source,cx.base,cx.core
ROOT=Path(__file__).resolve().parent
N,TOTAL=24567,28660
ARMS=['zero','random_half','neighbor_mean','query_mean','query_shuffle','query_flip']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)

def initialize(low,new,src,name):
    tick=time.monotonic();e,r=low['E'].double(),low['R'].double();random=e[N:];radius=random.norm(dim=-1,keepdim=True)
    # Query-space caches contain actual endpoint gradients for each model, no SVD rebuild.
    cache=torch.load(src/'cache/query-space.pt',map_location='cpu',weights_only=True)
    ids=cache['rows'];q=unit(cache['vectors']);qm=torch.zeros_like(random);qm.index_add_(0,ids,q)
    supported=qm.norm(dim=-1)>1e-30;query=random.clone();query[supported]=unit(qm[supported])*radius[supported]
    keys=set()
    for h,rel,t in new:
        if h<N<=t:keys.add((int(t)-N,int(h),int(rel),0 if name=='ComplEx' else 0))
        elif t<N<=h:keys.add((int(h)-N,int(t),int(rel),1 if name=='ComplEx' else 0))
    a=np.array(sorted(keys));rows,neighbors,relations,roles=a.T
    vec=unit(e[neighbors]);nm=torch.zeros_like(random);nm.index_add_(0,torch.as_tensor(rows),vec)
    neighbor=random.clone();ns=nm.norm(dim=-1)>1e-30;neighbor[ns]=unit(nm[ns])*radius[ns]
    ix=torch.nonzero(supported).flatten();p=np.random.default_rng(1220017).permutation(len(ix))
    shuffled=random.clone();shuffled[ix]=unit(query[ix[p]])*radius[ix]
    flipped=random.clone();flipped[supported]=-query[supported]
    variants=dict(zero=torch.zeros_like(random),random_half=random*.5,neighbor_mean=neighbor,query_mean=query,query_shuffle=shuffled,query_flip=flipped)
    report=dict(seconds=time.monotonic()-tick,unique_constraints=len(q),supported_rows=int(supported.sum()),unsupported_rows=int((~supported).sum()),
        mean_radius=float(radius.mean()),norm_error={k:float((v.norm(dim=-1)-radius[:,0]).abs().max()) for k,v in variants.items() if k not in ['zero','random_half']},
        query_alignment={k:float((unit(v[ids])*q).sum(-1).mean()) for k,v in variants.items()},query_cache_reused=True,head_tail_correct=(name=='ComplEx'))
    states={}
    for k,v in variants.items():
        states[k]=dict(E=low['E'].clone(),R=low['R']);states[k]['E'][N:]=v.float()
    return states,report

def train(name,arm,initial,new,tape,out):
    tick=time.monotonic();E,R,opt=source.model(initial);directory=out/'models'/arm;directory.mkdir(parents=True,exist_ok=True)
    torch.save(initial,directory/'initial.pt');last=None
    for j in range(len(tape['epochs'])):
        loss=cx.learning(E,R,new,tape,j) if name=='ComplEx' else base.learning(E,R,new,tape,j)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(E).all()
        if j==109:
            torch.save(dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict())),directory/'prefix10.pt');last=time.monotonic()-tick
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    report=dict(seconds=time.monotonic()-tick,prefix_seconds_included=last,steps=len(tape['epochs']),frozen_equal=True,zero_new_rows=int((E.detach()[N:].norm(dim=-1)==0).sum()))
    base.write(directory/'training.json',report);return report

def describe(teacher,states):
    eligible=teacher[:,1]<=10;d={k:eligible&(v[:,1]>10) for k,v in states.items()};order=list(states)
    return dict(phases={k:dict(drops=int(v.sum()),eligible=int(eligible.sum()),mrr=float((1/states[k][:,1]).mean())) for k,v in d.items()},
        transitions={f'{a}_to_{b}':dict(new_drops=int((eligible&~d[a]&d[b]).sum()),rescued=int((d[a]&~d[b]).sum())) for a,b in zip(order,order[1:])})

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    base.ROOT=core.ROOT=ROOT;base.N=core.N=N;base.TOTAL=core.TOTAL=TOTAL
    tick=time.monotonic();status('prepare');trainz=base.load('/data/lc/untzh/wn_transfer112_20261014/cache/train.npz');new=trainz['new'];truths=core.truthmap(trainz['old'],new)
    tape=base.load('/data/lc/untzh/wn_transfer112_20261014/cache/learning.npz');bundles=[]
    for name,folder,lowpath in [('DistMult','wn_transfer112_20261014','/data/lc/untzh/init_onset115_20261014/cache/low-initial.pt'),('ComplEx','complex_transfer118_20261014','/data/lc/untzh/complex_transfer118_20261014/cache/initial.pt')]:
        src=Path('/data/lc/untzh')/folder;out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
        low=torch.load(lowpath,map_location='cpu',weights_only=True);states,preparation=initialize(low,new,src,name);base.write(out/'initialization.json',preparation);training={}
        for arm in ARMS:status('train',model=name,arm=arm);training[arm]=train(name,arm,states[arm],new,tape,out)
        bundles.append((name,src,out,preparation,training))
    # All twelve new endpoints committed before current VALID evaluation.
    reports={}
    for name,src,out,preparation,training in bundles:
        status('evaluation_prepare',model=name);queries=base.load(src/'evaluation/queries.npz');cache={s:base.load(src/'cache'/f'fast-{s}.npz') for s in ['old_train','old_valid']}
        fast=cx.fast_ranks if name=='ComplEx' else source.fast_ranks;rank=cx.ranks if name=='ComplEx' else base.ranks
        controls=json.loads((src/'summary.json').read_text(encoding='utf8'))['metrics']
        if name=='DistMult':low_control=json.loads(Path('/data/lc/untzh/init_onset115_20261014/summary.json').read_text(encoding='utf8'))['metrics']['lowinit_none']
        else:low_control=controls['none']
        metrics={};phases={};ec={};evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True)
        for arm in ARMS:
            status('evaluate',model=name,arm=arm);p=out/'models'/arm
            init=torch.load(p/'initial.pt',map_location='cpu',weights_only=True);pref=torch.load(p/'prefix10.pt',map_location='cpu',weights_only=True);endpoint=torch.load(p/'epoch-20.pt',map_location='cpu',weights_only=True)
            E,R=endpoint['E'].cuda(),endpoint['R'].cuda();states={};cost={};stages={}
            for split in ['old_train','old_valid']:
                states[split],cost[split]=fast(E,R,queries[split],cache[split],truths)
                first,c0=fast(init['E'].cuda().double(),R,queries[split],cache[split],truths)
                stage=dict(old_candidates=cache[split]['teacher'],expanded_initial=first)
                if split=='old_valid':stage['epoch10'],c10=fast(pref['E'].cuda().double(),R,queries[split],cache[split],truths)
                else:c10=None
                stage['epoch20']=states[split];np.savez(evaluation/f'{arm}-{split}-phases.npz',**stage)
                stages[split]=describe(cache[split]['teacher'],stage);cost[split].update(initial_scoring=c0,prefix_scoring=c10)
            st=time.monotonic();states['new_valid']=rank(E,R,queries['new_valid'],truths);cost['new_valid']=dict(seconds=time.monotonic()-st);np.savez(evaluation/f'{arm}.npz',**states)
            m={split:base.metrics(states[split],q) for split,q in queries.items()}
            for split in ['old_train','old_valid']:m[split]['drops']=core.drops(states[split],cache[split]['teacher'])
            m['directional_ratios_to_low_none']=[m['new_valid'][d]['filtered_train_truths']['mrr']/low_control['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=m;phases[arm]=stages;ec[arm]=cost;base.write(evaluation/'metrics-running.json',metrics)
            print('EVALUATED122',name,arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_valid']['drops']['filtered_train_truths']['drops'],m['directional_ratios_to_low_none'],flush=True)
        reports[name]=dict(metrics=metrics,phases=phases,reused_low_control=low_control,reused_standard_controls=controls,preparation=preparation,training=training,evaluation=ec);base.write(out/'summary.json',reports[name])
    base.write(ROOT/'summary.json',dict(models=reports,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
