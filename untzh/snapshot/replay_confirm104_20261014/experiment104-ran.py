import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/anchor_rank103_20261014')
import experiment103 as previous
core,base=previous.core,previous.base
ROOT=Path(__file__).resolve().parent
N,TOTAL=previous.N,13134
SPECS={'replay_start0_w03':(0,'weight',False),
       'replay_late_w03':(220,'weight',False),
       'replay_late_rho01':(220,'norm',False),
       'anchor_replay_rho01':(220,'norm',True)}


def status(stage,**extra):
    base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**extra))


def replay_tape(seed,old,new,extra,nr,learning,out):
    tick=time.monotonic();rng=np.random.default_rng(1040000+seed)
    steps=len(learning['epochs']);negatives=learning['nh'].shape[1]
    pos=rng.integers(len(old),size=steps*32,dtype=np.int32);p=old[pos];h,r,t=p.T
    draw=rng.integers(TOTAL,size=(2,len(p),negatives),dtype=np.int32)
    tail=rng.random((len(p),negatives))>.5
    nh=np.where(tail,h[:,None],draw[0]).astype(np.int32)
    nt=np.where(tail,draw[1],t[:,None]).astype(np.int32)
    alltrain=np.concatenate([old,new,extra])
    known=np.unique((alltrain[:,0]*nr+alltrain[:,1])*TOTAL+alltrain[:,2])
    rejected=0;rounds=0
    while True:
        keys=(nh.astype(np.int64)*nr+r[:,None])*TOTAL+nt
        ix=np.searchsorted(known,keys);bad=(ix<len(known))&(known[np.minimum(ix,len(known)-1)]==keys)
        if not bad.any():break
        rejected+=int(bad.sum());rounds+=1;values=rng.integers(TOTAL,size=int(bad.sum()),dtype=np.int32)
        nh[bad]=np.where(tail[bad],nh[bad],values);nt[bad]=np.where(tail[bad],values,nt[bad])
    replay=dict(pos=pos,nh=nh,nt=nt,bounds=np.arange(steps+1)*32,epochs=learning['epochs'])
    (out/'cache').mkdir();np.savez(out/'cache/replay.npz',**replay)
    return replay,dict(seconds=time.monotonic()-tick,positive_rows=len(p),facts_per_step=32,negatives_per_fact=negatives,
                       bytes=sum(x.nbytes for x in replay.values()),rejections=rejected,rejection_rounds=rounds,seed=1040000+seed,
                       historical_pool=len(old),buffer='all historical TRAIN; not strict online fixed-size buffer')


def train(name,spec,initial,prefix,old,new,learning,replay,space,anchors,out):
    start,mode,project=spec;tick=time.monotonic()
    E,R,opt=core.newmodel(initial,prefix if start else None);records=[]
    for j in range(start,440):
        loss=base.learning(E,R,new,learning,j);gl=torch.autograd.grad(loss,E)[0]
        oldloss=base.learning(E,R,old,replay,j);gp=torch.autograd.grad(oldloss,E)[0]
        bn,pn=float(gl[N:].double().norm()),float(gp[N:].double().norm())
        alpha=.3 if mode=='weight' else .1*bn/pn if pn else 0.
        assert torch.isfinite(loss) and torch.isfinite(oldloss) and torch.isfinite(gl+alpha*gp).all()
        opt.zero_grad(set_to_none=True);E.grad=gl+alpha*gp;opt.step()
        displacement_norm=None;max_change=None
        if project:
            reference=anchors[j-220].cuda();raw=(E.detach()[N:]-reference).double();accepted=previous.prior.project(raw,space)
            assert float(accepted.norm())<=float(raw.norm())*(1+1e-8)+1e-12
            with torch.no_grad():E[N:].copy_(reference+accepted.to(E.dtype))
            actual=(E.detach()[N:]-reference).double();changes=(space['vectors']*actual[space['rows']]).sum(-1)
            displacement_norm=float(actual.norm());max_change=float(changes.abs().max())
        assert torch.isfinite(E).all()
        records.append(dict(step=j,epoch=int(learning['epochs'][j]),alpha=alpha,gradient_ratio=alpha*pn/bn if bn else 0.,
                            replay_gradient_nonzero=pn>0,anchor_displacement_norm=displacement_norm,
                            max_cross_train_score_difference=max_change))
        if j==439 or learning['epochs'][j+1]!=learning['epochs'][j]:core.progress('train',arm=name,epoch=int(learning['epochs'][j]))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    directory=out/'models'/name;directory.mkdir(parents=True)
    torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),directory/'epoch-20.pt')
    base.write(directory/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,steps=len(records),frozen_equal=True,
                mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in records])),max_gradient_ratio=max(x['gradient_ratio'] for x in records),
                nonzero_replay_steps=sum(x['replay_gradient_nonzero'] for x in records),
                final_anchor_displacement_norm=records[-1]['anchor_displacement_norm'],
                max_cross_train_score_difference=max(x['max_cross_train_score_difference'] for x in records) if project else None)
    base.write(directory/'training.json',report);return report


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();status('starting');z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    old,new,extra=z['old'],z['new'],z['extra'];truths=core.truthmap(old,np.concatenate([new,extra]));bundles=[]
    for seed in [29,17]:
        out=ROOT/f'seed{seed}';out.mkdir()
        training_source=Path('/data/lc/untzh')/('risk_increment88_20261014' if seed==29 else 'increment87_20261014')
        previous.configure(out,training_source);torch.manual_seed(seed);status('prepare',seed=seed);prep=time.monotonic()
        initial=torch.load(training_source/'cache/initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(training_source/'cache/prefix10.pt',map_location='cpu',weights_only=True)
        learning=base.load(training_source/'cache/learning.npz');assert len(learning['epochs'])==440
        replay,rc=replay_tape(seed,old,new,extra,len(initial['R']),learning,out)
        anchor_source=Path('/data/lc/untzh')/('anchor_space94_20261014' if seed==29 else 'anchor_confirm95_20261014')
        space_source=Path('/data/lc/untzh')/('query_space92_20261014' if seed==29 else 'anchor_confirm95_20261014')
        space=previous.load_space(space_source/'cache/query-space.pt')
        anchors=torch.load(anchor_source/'cache/ordinary-anchors.pt',map_location='cpu',weights_only=True)
        costs=dict(preparation_seconds=time.monotonic()-prep,replay_tape=rc,historical_assets_not_free=True,training={})
        for name,spec in SPECS.items():
            status('train',seed=seed,arm=name)
            costs['training'][name]=train(name,spec,initial,prefix,old,new,learning,replay,space,anchors,out)
            base.write(out/'costs-running.json',costs)
        bundles.append((seed,out,training_source,costs))
    summaries={}
    for seed,out,training_source,costs in bundles:
        previous.configure(out,training_source)
        status('evaluate',seed=seed)
        metrics,seconds,reference=previous.evaluate(seed,out,training_source,truths,list(SPECS))
        costs['evaluation_seconds']=seconds
        summaries[str(seed)]=dict(metrics=metrics,controls_full_valid=reference,costs=costs)
        base.write(out/'summary.json',summaries[str(seed)])
    base.write(ROOT/'summary.json',dict(seeds=summaries,total_seconds=time.monotonic()-tick))
    status('complete',seconds=time.monotonic()-tick);print('COMPLETE104',time.monotonic()-tick,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
