import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import collections
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/anchor_rank103_20261014')
import experiment103 as previous
core,base=previous.core,previous.base
ROOT=Path(__file__).resolve().parent
previous.ROOT=ROOT
N,TOTAL=previous.N,13134
ARMS=['full_replay_joint','full_replay_anchor']


def status(stage,**extra):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**extra))


def prepare_replay(seed,initial,old,truths,heads,out):
    tick=time.monotonic()
    replay=base.load(f'/data/lc/untzh/replay_confirm104_20261014/seed{seed}/cache/replay.npz')
    positions=replay['pos'][replay['bounds'][220]:replay['bounds'][440]];p=old[positions]
    assert len(p)==220*32
    qtail=(initial['E'][p[:,0]].double()*initial['R'][p[:,1]].double()).numpy().reshape(220,32,200)
    qhead=(initial['E'][p[:,2]].double()*initial['R'][p[:,1]].double()).numpy().reshape(220,32,200)
    mt=np.ones((len(p),TOTAL-N),bool);mh=np.ones_like(mt);dt=np.empty(len(p));dh=np.empty(len(p))
    for i,(h,r,t) in enumerate(p):
        ts=truths[int(h),int(r)];hs=heads[int(r),int(t)]
        assert int(t) in ts and int(h) in hs
        dt[i]=TOTAL-len(ts);dh[i]=TOTAL-len(hs)
        mt[i,[x-N for x in ts if x>=N]]=False;mh[i,[x-N for x in hs if x>=N]]=False
    assert (dt>0).all() and (dh>0).all()
    payload=dict(qtail=qtail,qhead=qhead,mask_tail=mt.reshape(220,32,-1),mask_head=mh.reshape(220,32,-1),
                 denominator_tail=dt.reshape(220,32),denominator_head=dh.reshape(220,32),positions=positions)
    (out/'cache').mkdir();np.savez(out/'cache/full-replay.npz',**payload)
    data={k:torch.as_tensor(v,device='cuda') for k,v in payload.items() if k!='positions'}
    return data,dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in payload.values()),
                     historical_facts=32,steps=220,pool='all historical TRAIN, same sampled positives as104',
                     all_candidates=TOTAL,computed_new_candidates=TOTAL-N,old_terms_omitted='constant under frozen old E/R')


def penalty(E,data,k):
    tail=F.softplus(data['qtail'][k]@E[N:].double().T).masked_fill(~data['mask_tail'][k],0).sum(-1)/data['denominator_tail'][k]
    head=F.softplus(data['qhead'][k]@E[N:].double().T).masked_fill(~data['mask_head'][k],0).sum(-1)/data['denominator_head'][k]
    return .5*(tail+head).mean()


def train(arm,initial,prefix,new,tape,data,space,anchors,out):
    tick=time.monotonic();E,R,opt=core.newmodel(initial,prefix);records=[]
    for j in range(220,440):
        loss=base.learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0]
        p=penalty(E,data,j-220);gp=torch.autograd.grad(p,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm());alpha=.1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(p) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        reference=anchors[j-220].cuda();difference=(E.detach()[N:]-reference).double()
        if arm=='full_replay_anchor':
            projected=previous.prior.project(difference,space)
            assert float(projected.norm())<=float(difference.norm())*(1+1e-8)+1e-12
            with torch.no_grad():E[N:].copy_(reference+projected.to(E.dtype))
            difference=(E.detach()[N:]-reference).double()
        changes=(space['vectors']*difference[space['rows']]).sum(-1);assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=int(tape['epochs'][j]),gradient_ratio=alpha*pn/bn if bn else 0.,
                            anchor_displacement_norm=float(difference.norm()),max_cross_train_score_difference=float(changes.abs().max())))
        if j==439 or tape['epochs'][j+1]!=tape['epochs'][j]:core.progress('train',arm=arm,epoch=int(tape['epochs'][j]))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    directory=out/'models'/arm;directory.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,steps=220,frozen_equal=True,mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in records])),
                final_anchor_displacement_norm=records[-1]['anchor_displacement_norm'],
                max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records))
    base.write(directory/'training.json',report);return report


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();status('starting');z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    old,new,extra=z['old'],z['new'],z['extra'];truths=core.truthmap(old,np.concatenate([new,extra]))
    heads=collections.defaultdict(set)
    for h,r,t in np.concatenate([old,new,extra]):heads[int(r),int(t)].add(int(h))
    bundles=[]
    for seed in [29,17]:
        out=ROOT/f'seed{seed}';out.mkdir()
        source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        previous.configure(out,source);torch.manual_seed(seed);status('prepare',seed=seed);prep=time.monotonic()
        initial=torch.load(source/'cache/initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(source/'cache/prefix10.pt',map_location='cpu',weights_only=True);tape=base.load(source/'cache/learning.npz')
        data,rc=prepare_replay(seed,initial,old,truths,heads,out)
        anchor=Path('/data/lc/untzh')/('anchor_space94_20261014' if seed==29 else 'anchor_confirm95_20261014')
        spaces=Path('/data/lc/untzh')/('query_space92_20261014' if seed==29 else 'anchor_confirm95_20261014')
        space=previous.load_space(spaces/'cache/query-space.pt')
        anchors=torch.load(anchor/'cache/ordinary-anchors.pt',map_location='cpu',weights_only=True)
        costs=dict(preparation_seconds=time.monotonic()-prep,full_replay_cache=rc,historical_assets_not_free=True,training={})
        for arm in ARMS:
            status('train',seed=seed,arm=arm);costs['training'][arm]=train(arm,initial,prefix,new,tape,data,space,anchors,out)
            base.write(out/'costs-running.json',costs)
        bundles.append((seed,out,source,costs))
    summaries={}
    for seed,out,source,costs in bundles:
        previous.configure(out,source);status('evaluate',seed=seed)
        metrics,seconds,reference=previous.evaluate(seed,out,source,truths,ARMS);costs['evaluation_seconds']=seconds
        summaries[str(seed)]=dict(metrics=metrics,controls_full_valid=reference,costs=costs);base.write(out/'summary.json',summaries[str(seed)])
    base.write(ROOT/'summary.json',dict(seeds=summaries,total_seconds=time.monotonic()-tick))
    status('complete',seconds=time.monotonic()-tick);print('COMPLETE105',time.monotonic()-tick,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
