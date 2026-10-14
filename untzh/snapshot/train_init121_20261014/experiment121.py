import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
import copy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/lowinit_confirm116_20261014')
import experiment116 as previous
online,core,base,pair=previous.online,previous.core,previous.base,previous.pair
ROOT=Path(__file__).resolve().parent
N,TOTAL=11258,13134
ARMS=['zero','random_half','neighbor_mean','query_mean','query_shuffle','query_flip']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)

def initializations(initial,new,seed):
    tick=time.monotonic();e,r=initial['E'].double(),initial['R'].double()
    low=e[N:].clone();radius=low.norm(dim=-1,keepdim=True);keys=set()
    for h,rel,t in new:
        if h<N<=t:keys.add((int(t)-N,int(h),int(rel),0))
        elif t<N<=h:keys.add((int(h)-N,int(t),int(rel),1))
    a=np.array(sorted(keys));ids,neighbors,relations,roles=a.T
    # Both endpoint gradients for DistMult are neighbor*relation; inverse R kept distinct.
    q=unit(e[neighbors]*r[relations]);v=unit(e[neighbors]);counts=torch.bincount(torch.as_tensor(ids),minlength=TOTAL-N)
    qm=torch.zeros_like(low);nm=torch.zeros_like(low)
    qm.index_add_(0,torch.as_tensor(ids),q);nm.index_add_(0,torch.as_tensor(ids),v)
    qsupported=qm.norm(dim=-1)>1e-30;nsupported=nm.norm(dim=-1)>1e-30
    qinit=low.clone();ninit=low.clone();qinit[qsupported]=unit(qm[qsupported])*radius[qsupported];ninit[nsupported]=unit(nm[nsupported])*radius[nsupported]
    shuffle=low.clone();indices=torch.flatnonzero(qsupported) if hasattr(torch,'flatnonzero') else torch.nonzero(qsupported).flatten()
    perm=np.random.default_rng(1210000+seed).permutation(len(indices))
    shuffle[indices]=unit(qinit[indices[perm]])*radius[indices]
    flipped=low.clone();flipped[qsupported]=-qinit[qsupported]
    variants=dict(zero=torch.zeros_like(low),random_half=low*.5,neighbor_mean=ninit,query_mean=qinit,query_shuffle=shuffle,query_flip=flipped)
    report=dict(seconds=time.monotonic()-tick,unique_role_constraints=len(a),supported_rows=int(qsupported.sum()),unsupported_rows=int((~qsupported).sum()),
                cross_degree_mean=float(counts.double().mean()),mean_low_radius=float(radius.mean()),
                norm_match_max_error={k:float((x.norm(dim=-1)-radius[:,0]).abs().max()) for k,x in variants.items() if k not in ['zero','random_half']},
                query_alignment={k:float((unit(x[ids])*q).sum(-1).mean()) for k,x in variants.items()},
                queries_from_new_train_only=True,unsupported_fallback='same low random vector')
    states={}
    for name,rows in variants.items():
        z=dict(E=initial['E'].clone(),R=initial['R']);z['E'][N:]=rows.float();states[name]=z
    return states,report

def train(arm,initial,new,tape,out):
    torch.cuda.synchronize();tick=time.monotonic();E,R,opt=core.newmodel(initial)
    directory=out/'models'/arm;directory.mkdir(parents=True,exist_ok=True)
    torch.save(initial,directory/'initial.pt');prefix_seconds=None
    for j in range(len(tape['epochs'])):
        loss=base.learning(E,R,new,tape,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        assert torch.isfinite(E).all()
        if j==219:
            prefix=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()))
            torch.save(prefix,directory/'prefix10.pt');prefix_seconds=time.monotonic()-tick
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    torch.cuda.synchronize()
    report=dict(seconds=time.monotonic()-tick,prefix_seconds_included=prefix_seconds,steps=len(tape['epochs']),frozen_equal=True,
                final_zero_new_rows=int((E.detach()[N:].norm(dim=-1)==0).sum()))
    base.write(directory/'training.json',report);return report

def transitions(teacher,states):
    eligible=teacher[:,1]<=10;d={k:eligible&(v[:,1]>10) for k,v in states.items()};out={}
    for k,mask in d.items():out[k]=dict(drops=int(mask.sum()),eligible=int(eligible.sum()),mrr=float((1/states[k][:,1]).mean()))
    order=list(states);changes={}
    for before,after in zip(order,order[1:]):changes[f'{before}_to_{after}']=dict(new_drops=int((eligible&~d[before]&d[after]).sum()),rescued=int((d[before]&~d[after]).sum()))
    return dict(phases=out,transitions=changes)

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();status('prepare');z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz');new=z['new']
    truths=core.truthmap(z['old'],np.concatenate([new,z['extra']]));bundles=[]
    for seed in [43,17,29]:
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',17:'increment87_20261014',29:'risk_increment88_20261014'}[seed]
        old=Path(f'/data/lc/untzh/lowinit_confirm116_20261014/seed{seed}');out=ROOT/f'seed{seed}'
        out.mkdir(parents=True,exist_ok=True);online.configure(out,src);tape=base.load(src/'cache/learning.npz')
        low=torch.load(old/'cache/initial.pt',map_location='cpu',weights_only=True)
        variants,preparation=initializations(low,new,seed);base.write(out/'initialization.json',preparation);training={}
        for arm in ARMS:
            status('train',seed=seed,arm=arm);training[arm]=train(arm,variants[arm],new,tape,out)
        bundles.append((seed,src,old,out,low,preparation,training))
    # All eighteen ordinary endpoints committed before current VALID evaluation.
    full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz');reports={}
    for seed,src,old,out,low,preparation,training in bundles:
        online.configure(out,src);status('evaluation_prepare',seed=seed)
        fast=pair.fast43(low,truths) if seed==43 else pair.fastmodule.FastOld(seed,truths)
        queries=dict(fast.queries,new_valid=full['new_valid']);metrics={};phase_reports={};ec={}
        reference=json.loads((old/'summary.json').read_text(encoding='utf8'))['metrics'];directory=out/'evaluation';directory.mkdir(exist_ok=True)
        for arm in ARMS:
            status('evaluate',seed=seed,arm=arm);path=out/'models'/arm
            init=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
            prefix=torch.load(path/'prefix10.pt',map_location='cpu',weights_only=True)
            endpoint=torch.load(path/'epoch-20.pt',map_location='cpu',weights_only=True)
            E,R=endpoint['E'].cuda(),endpoint['R'].cuda();states={};cost={};phases={}
            for split in ['old_train','old_valid']:
                states[split],cost[split]=fast.ranks(E,R,split)
                initial_rank,c0=fast.ranks(init['E'].cuda().double(),R,split)
                stage=dict(old_candidates=fast.teacher[split],expanded_initial=initial_rank)
                if split=='old_valid':stage['epoch10'],c10=fast.ranks(prefix['E'].cuda().double(),R,split)
                else:c10=None
                stage['epoch20']=states[split];np.savez(directory/f'{arm}-{split}-phases.npz',**stage)
                phases[split]=transitions(fast.teacher[split],stage);cost[split].update(initial_scoring=c0,prefix_scoring=c10)
            st=time.monotonic();states['new_valid']=base.ranks(E,R,full['new_valid'],truths,arm=arm,split='new_valid');cost['new_valid']=dict(seconds=time.monotonic()-st)
            np.savez(directory/f'{arm}.npz',**states);m={s:base.metrics(states[s],q) for s,q in queries.items()}
            for split in ['old_train','old_valid']:m[split]['drops']=core.drops(states[split],fast.teacher[split])
            m['directional_ratios_to_low_none']=[m['new_valid'][d]['filtered_train_truths']['mrr']/reference['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
            metrics[arm]=m;phase_reports[arm]=phases;ec[arm]=cost
            base.write(directory/'metrics-running.json',metrics);base.write(directory/'phases-running.json',phase_reports)
            print('EVALUATED121',seed,arm,m['new_valid']['all']['filtered_train_truths']['mrr'],m['old_train']['drops']['filtered_train_truths']['drops'],m['old_valid']['drops']['filtered_train_truths']['drops'],m['directional_ratios_to_low_none'],flush=True)
        reports[str(seed)]=dict(metrics=metrics,phases=phase_reports,reused_low_controls=reference,initialization=preparation,training=training,evaluation=ec)
        base.write(out/'summary.json',reports[str(seed)])
    base.write(ROOT/'summary.json',dict(seeds=reports,total_seconds=time.monotonic()-tick));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
