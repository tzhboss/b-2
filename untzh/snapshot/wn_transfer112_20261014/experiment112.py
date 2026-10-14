import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/pair_constraint111_20261014')
import experiment111 as pairmodule
base,core=pairmodule.base,pairmodule.core
space_module=pairmodule.online.prior
ROOT=Path(__file__).resolve().parent
DATA=Path('/data/lc/tzh/experiment4/literature_baselines_20261010/repos/FastKGE/data/WN_CKGE')
N,TOTAL,D,SEED=24567,28660,200,17
ARMS=['none','absolute','answer_anchor','query_anchor','query_pair4','full_replay_joint','full_replay_anchor']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def mapping(ep,name):return {a:int(b) for a,b in (s.split() for s in (DATA/str(ep)/name).read_text().splitlines())}
def triples(ep,split):
    e=mapping(ep,'entity2id.txt');r=mapping(ep,'relation2id.txt');out=[]
    for h,rr,t in (s.split() for s in (DATA/str(ep)/(split+'.txt')).read_text().splitlines()):
        h,rr,t=e[h],r[rr],e[t];assert rr%2==0;out.extend([(h,rr,t),(t,rr+1,h)])
    return np.asarray(out,dtype=np.int64)

def tape(facts,truths,total,nr,epochs,seed,name):
    tick=time.monotonic();rng=np.random.default_rng(seed);known=np.sort(np.array([(h*nr+r)*total+t for (h,r),ts in truths.items() for t in ts],dtype=np.int64))
    pieces={k:[] for k in ['pos','nh','nt']};bounds=[0];epids=[];rejections=0
    for ep in range(epochs):
        ids=rng.permutation(len(facts)).astype(np.int32);p=facts[ids];h,r,t=p.T
        draw=rng.integers(total,size=(2,len(p),10),dtype=np.int32);tail=rng.random((len(p),10))>.5
        nh=np.where(tail,h[:,None],draw[0]).astype(np.int32);nt=np.where(tail,draw[1],t[:,None]).astype(np.int32)
        while True:
            keys=(nh.astype(np.int64)*nr+r[:,None])*total+nt;ix=np.searchsorted(known,keys)
            bad=(ix<len(known))&(known[np.minimum(ix,len(known)-1)]==keys)
            if not bad.any():break
            rejections+=int(bad.sum());values=rng.integers(total,size=int(bad.sum()),dtype=np.int32)
            nh[bad]=np.where(tail[bad],nh[bad],values);nt[bad]=np.where(tail[bad],values,nt[bad])
        pieces['pos'].append(ids);pieces['nh'].append(nh);pieces['nt'].append(nt)
        for lo in range(0,len(p),1024):bounds.append(bounds[-1]+min(1024,len(p)-lo));epids.append(ep+1)
    result={k:np.concatenate(v) for k,v in pieces.items()};result.update(bounds=np.array(bounds),epochs=np.array(epids))
    np.savez(ROOT/'cache'/f'{name}.npz',**result)
    return result,dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in result.values()),rejections=rejections,seed=seed,steps=len(epids),epochs=epochs)

def model(initial,prefix=None):
    E=torch.nn.Parameter(initial['E'].cuda().clone());R=initial['R'].cuda().clone()
    E.register_hook(lambda g:torch.cat([torch.zeros_like(g[:N]),g[N:]]))
    opt=torch.optim.Adam([E],lr=.007)
    if prefix:E.data.copy_(prefix['E'].cuda());opt.load_state_dict(copy.deepcopy(prefix['optimizer']))
    return E,R,opt

def save(E,R,arm):
    directory=ROOT/'models'/arm;directory.mkdir(parents=True);torch.save(dict(E=E.detach().cpu().double(),R=R.detach().cpu().double()),directory/'epoch-20.pt')

@torch.no_grad()
def teacher_cache(initial,queries,truths,split):
    tick=time.monotonic();E=initial['E'][:N].double().cuda();R=initial['R'].double().cuda();ranks=[];gold=[];newrows=[];newcols=[]
    for lo in range(0,len(queries),64):
        b=queries[lo:lo+64];q=torch.as_tensor(b,device='cuda');scores=(E[q[:,0]]*R[q[:,1]])@E.T
        gs=scores[torch.arange(len(b),device='cuda'),q[:,2]].clone();raw=1+(scores>gs[:,None]).sum(1);rows=[];cols=[]
        for j,(h,r,g) in enumerate(b):
            for c in truths[int(h),int(r)]:
                if c>=N:newrows.append(lo+j);newcols.append(c-N)
                elif c!=int(g):rows.append(j);cols.append(c)
        if rows:scores[rows,cols]=-torch.inf
        ranks.append(torch.stack([raw,1+(scores>gs[:,None]).sum(1)],1).cpu().numpy());gold.append(gs.cpu().numpy())
    z=dict(teacher=np.concatenate(ranks),gold=np.concatenate(gold),rows=np.array(newrows,dtype=np.int64),cols=np.array(newcols,dtype=np.int64))
    np.savez(ROOT/'cache'/f'fast-{split}.npz',**z)
    return z,dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in z.values()),eligible_filtered=int((z['teacher'][:,1]<=10).sum()))

def fast_ranks(E,R,queries,z,truths):
    tick=time.monotonic();parts=[];fallback=0
    for lo in range(0,len(queries),64):
        b=queries[lo:lo+64];q=torch.as_tensor(b,device='cuda');gold=torch.as_tensor(z['gold'][lo:lo+len(b)],device='cuda')
        s=(E[q[:,0]]*R[q[:,1]])@E[N:].T
        if ((s-gold[:,None]).abs()<=1e-10*(1+gold.abs()[:,None])).any():parts.append(base.ranks(E,R,b,truths));fallback+=len(b);continue
        raw=(s>gold[:,None]).sum(1).cpu().numpy();i=np.searchsorted(z['rows'],lo);j=np.searchsorted(z['rows'],lo+len(b))
        if j>i:s[z['rows'][i:j]-lo,z['cols'][i:j]]=-torch.inf
        filtered=(s>gold[:,None]).sum(1).cpu().numpy();parts.append(z['teacher'][lo:lo+len(b)]+np.column_stack([raw,filtered]))
    return np.concatenate(parts),dict(seconds=time.monotonic()-tick,fallback_queries=fallback)

def protection(E,data,chosen,full):
    if not full:return F.softplus((data['q'][chosen,None]*E[data['candidates'][chosen]].double()).sum(-1)).mean()
    s=data['q'][chosen]@E[N:].double().T;legal=data['legal'][chosen]
    return (F.softplus(s).masked_fill(~legal,0).sum(-1)/legal.sum(-1)).mean()

def replay_penalty(E,data,k):
    tail=F.softplus(data['qt'][k]@E[N:].double().T).masked_fill(~data['mt'][k],0).sum(-1)/data['dt'][k]
    head=F.softplus(data['qh'][k]@E[N:].double().T).masked_fill(~data['mh'][k],0).sum(-1)/data['dh'][k]
    return .5*(tail+head).mean()

def train(arm,initial,prefix,new,t,data,space,replay,calibration,start):
    E,R,opt=model(initial,prefix);shadow=None
    if arm not in ['none','absolute']:shadow=model(initial,prefix)
    tick=time.monotonic();rng=np.random.default_rng(1010029);records=[];correction_seconds=0.
    pairs=new[(new[:,0]>=N)&(new[:,2]>=N)];pp=torch.as_tensor(pairs[:,[0,2]]-N,device='cuda');rr=R[pairs[:,1]].double();degree=torch.bincount(pp.flatten(),minlength=TOTAL-N).clamp_min(1).double()
    for j in range(start,len(t['epochs'])):
        k=j-start;loss=base.learning(E,R,new,t,j);gl=torch.autograd.grad(loss,E)[0];bn=float(gl[N:].double().norm());alpha=0.;pn=0.
        if arm=='none':gradient=gl
        else:
            if arm.startswith('full_replay'):p=replay_penalty(E,replay,k)
            else:
                chosen=data['representatives'][rng.choice(len(data['representatives']),32,replace=False)] if arm in ['query_anchor','query_pair4'] else data['selections'][k]
                p=protection(E,data,chosen,arm!='absolute')
            gp=torch.autograd.grad(p,E)[0];pn=float(gp[N:].double().norm())
            alpha=min(calibration,.1*bn/pn) if arm=='absolute' and pn else .1*bn/pn if pn else 0.;gradient=gl+alpha*gp
        opt.zero_grad(set_to_none=True);E.grad=gradient;opt.step();cross=None;pair_rms=None;norm=None
        if shadow:
            se,sr,so=shadow;sl=base.learning(se,sr,new,t,j);so.zero_grad(set_to_none=True);sl.backward();so.step();reference=se.detach()[N:]
            delta=(E.detach()[N:]-reference).double()
            if arm!='full_replay_joint':
                delta=space_module.project(delta,space)
                if arm=='query_pair4':
                    st=time.monotonic();x,stats=pairmodule.correct(reference.double()+delta,reference.double(),pp,rr,degree,space,4);correction_seconds+=time.monotonic()-st;delta=x-reference.double()
                with torch.no_grad():E[N:].copy_(reference+delta.to(E.dtype))
            actual=(E.detach()[N:]-reference).double();cross=float((space['vectors']*actual[space['rows']]).sum(-1).abs().max())
            pair_rms=float((pairmodule.pair_scores(E.detach()[N:].double(),pp,rr)-pairmodule.pair_scores(reference.double(),pp,rr)).square().mean().sqrt());norm=float(actual.norm())
        assert torch.isfinite(E).all();records.append(dict(step=j,gradient_ratio=alpha*pn/bn if bn else 0.,cross_max=cross,pair_rms=pair_rms,norm=norm))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    if shadow:
        ordinary=torch.load(ROOT/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
        assert torch.equal(shadow[0].detach().cpu().double(),ordinary['E']) and torch.equal(shadow[1].cpu().double(),ordinary['R'])
    save(E,R,arm);base.write(ROOT/'models'/arm/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,steps=len(records),mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in records])),frozen_equal=True,shadow_compute_included=bool(shadow),correction_seconds_included=correction_seconds,final_norm=records[-1]['norm'],final_pair_rms=records[-1]['pair_rms'])
    base.write(ROOT/'models'/arm/'training.json',report);return report

def main():
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();base.ROOT=core.ROOT=space_module.ROOT=ROOT;core.N=base.N=space_module.N=N;core.TOTAL=base.TOTAL=space_module.TOTAL=TOTAL
    status('prepare');(ROOT/'cache').mkdir();old=triples(0,'train');new=triples(1,'train')
    e0,e1=mapping(0,'entity2id.txt'),mapping(1,'entity2id.txt');assert all(e1[a]==b for a,b in e0.items()) and mapping(0,'relation2id.txt')==mapping(1,'relation2id.txt')
    assert ((new[:,0]>=N)|(new[:,2]>=N)).all();truths=core.truthmap(old,new);oldtruths=core.truthmap(old,np.empty((0,3),np.int64));np.savez(ROOT/'cache/train.npz',old=old,new=new)
    costs={};pt,costs['parent_tape']=tape(old,oldtruths,N,22,50,1120017,'parent-learning');status('parent_train')
    E=torch.nn.Parameter(torch.randn((N,D),device='cuda')*.12);R=torch.nn.Parameter(torch.randn((22,D),device='cuda')*.12);opt=torch.optim.Adam([E,R],lr=.007);st=time.monotonic()
    for j in range(len(pt['epochs'])):
        loss=base.learning(E,R,old,pt,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(E).all() and torch.isfinite(R).all()
        if j==len(pt['epochs'])-1 or pt['epochs'][j+1]!=pt['epochs'][j]:status('parent_train',epoch=int(pt['epochs'][j]))
    costs['parent_training_seconds']=time.monotonic()-st;parent=dict(E=E.detach().cpu(),R=R.detach().cpu());torch.save(parent,ROOT/'cache/parent.pt');del pt,E,R,opt
    expanded=torch.empty((TOTAL,D),device='cuda');torch.nn.init.normal_(expanded,std=.12,generator=torch.Generator(device='cuda').manual_seed(1087));expanded[:N]=parent['E'].cuda();initial=dict(E=expanded.cpu(),R=parent['R']);torch.save(initial,ROOT/'cache/initial.pt')
    t,costs['increment_tape']=tape(new,truths,TOTAL,22,20,1120088,'learning');start=int(np.flatnonzero(t['epochs']==11)[0]);assert start*2==len(t['epochs'])
    status('teacher_train');traincache,costs['teacher_train']=teacher_cache(initial,old,truths,'old_train')
    st=time.monotonic();eligible=np.flatnonzero(traincache['teacher'][:,1]<=10);size=min(16384,len(eligible));assert size>=32
    rng=np.random.default_rng(SEED);ids=rng.choice(eligible,size,replace=False);queries=old[ids];candidates=rng.integers(N,TOTAL,size=(size,128));legal=np.ones((size,TOTAL-N),bool)
    for i,(h,r,g) in enumerate(queries):
        known=list(truths[int(h),int(r)]);bad=np.isin(candidates[i],known)
        while bad.any():candidates[i,bad]=rng.integers(N,TOTAL,size=int(bad.sum()));bad=np.isin(candidates[i],known)
        legal[i,[x-N for x in known if x>=N]]=False
    q=initial['E'][queries[:,0]].double()*initial['R'][queries[:,1]].double();groups,reps,inverse=np.unique(queries[:,:2],axis=0,return_index=True,return_inverse=True)
    selections=np.stack([np.random.default_rng(SEED+1000+j).choice(size,32,replace=False) for j in range(len(t['epochs'])-start)])
    np.savez(ROOT/'cache/cohort.npz',ids=ids,queries=queries,q=q.numpy(),candidates=candidates,legal=legal,representatives=reps,inverse=inverse,selections=selections)
    data=dict(q=q.cuda(),candidates=torch.as_tensor(candidates,device='cuda'),legal=torch.as_tensor(legal,device='cuda'),representatives=reps,selections=selections)
    costs['cohort']=dict(seconds=time.monotonic()-st,eligible=len(eligible),size=size,query_groups=len(groups))
    E,R,opt=model(initial);st=time.monotonic()
    for j in range(start):loss=base.learning(E,R,new,t,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    prefix=dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict()));torch.save(prefix,ROOT/'cache/prefix10.pt');costs['prefix_seconds']=time.monotonic()-st
    norms=[]
    for k in range(8):
        gl=torch.autograd.grad(base.learning(E,R,new,t,start+k),E)[0];gp=torch.autograd.grad(protection(E,data,selections[k],False),E)[0];norms.append([float(gl[N:].double().norm()),float(gp[N:].double().norm())])
    a=np.array(norms);calibration=.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2));base.write(ROOT/'calibration.json',dict(coefficient=calibration,norms=norms))
    space,costs['space']=space_module.build_space(initial,new)
    st=time.monotonic();count=len(t['epochs'])-start;positions=np.random.default_rng(1040017).integers(len(old),size=len(t['epochs'])*32);p=old[positions[start*32:]];heads={}
    for h,r,g in np.concatenate([old,new]):heads.setdefault((int(r),int(g)),set()).add(int(h))
    qt=(initial['E'][p[:,0]].double()*initial['R'][p[:,1]].double()).numpy().reshape(count,32,D);qh=(initial['E'][p[:,2]].double()*initial['R'][p[:,1]].double()).numpy().reshape(count,32,D)
    mt=np.ones((len(p),TOTAL-N),bool);mh=np.ones_like(mt);dt=np.empty(len(p));dh=np.empty(len(p))
    for i,(h,r,g) in enumerate(p):
        ts=truths[int(h),int(r)];hs=heads[int(r),int(g)];mt[i,[x-N for x in ts if x>=N]]=False;mh[i,[x-N for x in hs if x>=N]]=False;dt[i]=TOTAL-len(ts);dh[i]=TOTAL-len(hs)
    replaycache=dict(qt=qt,qh=qh,mt=mt.reshape(count,32,-1),mh=mh.reshape(count,32,-1),dt=dt.reshape(count,32),dh=dh.reshape(count,32));np.savez(ROOT/'cache/full-replay.npz',positions=positions,**replaycache);replay={k:torch.as_tensor(v,device='cuda') for k,v in replaycache.items()};costs['full_replay_cache']=dict(seconds=time.monotonic()-st,bytes=sum(v.nbytes for v in replaycache.values()))
    costs['preparation_seconds']=time.monotonic()-tick;costs['training']={}
    for arm in ARMS:status('train',arm=arm);costs['training'][arm]=train(arm,initial,prefix,new,t,data,space,replay,calibration,start);base.write(ROOT/'costs-running.json',costs)
    # All seven committed before any VALID read.
    status('evaluation_prepare');oldvalid=triples(0,'valid');sourcevalid=triples(1,'valid');mask=(sourcevalid[:,0]>=N)|(sourcevalid[:,2]>=N);newvalid=sourcevalid[mask]
    base.write(ROOT/'data.json',dict(old_entities=N,total_entities=TOTAL,relations=22,old_train=len(old),new_train=len(new),old_valid=len(oldvalid),new_valid=len(newvalid),source_new_valid=len(sourcevalid),excluded_old_old_valid=int((~mask).sum()),new_directions={d:int(((newvalid[:,0]<N)&(newvalid[:,2]>=N)).sum()) if i==0 else int(((newvalid[:,0]>=N)&(newvalid[:,2]<N)).sum()) if i==1 else int(((newvalid[:,0]>=N)&(newvalid[:,2]>=N)).sum()) for i,d in enumerate(core.DIRECTIONS)}))
    validcache,costs['teacher_valid']=teacher_cache(initial,oldvalid,truths,'old_valid');queries=dict(old_train=old,old_valid=oldvalid,new_valid=newvalid);zs=dict(old_train=traincache,old_valid=validcache)
    checks=[];fast_ok=True
    for arm in ['none','query_anchor']:
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);ee,rr=m['E'].cuda(),m['R'].cuda()
        for split in ['old_train','old_valid']:
            state,_=fast_ranks(ee,rr,queries[split],zs[split],truths)
            for lo in [0,(len(queries[split])//2//64)*64,((len(queries[split])-64)//64)*64]:
                exact=base.ranks(ee,rr,queries[split][lo:lo+64],truths);equal=np.array_equal(state[lo:lo+64],exact);fast_ok&=equal;checks.append(dict(arm=arm,split=split,start=lo,equal=equal))
    base.write(ROOT/'fast-spot-checks.json',dict(all_equal=fast_ok,checks=checks))
    out=ROOT/'evaluation';out.mkdir();np.savez(out/'queries.npz',**queries);np.savez(out/'teacher.npz',**{s:z['teacher'] for s,z in zs.items()});metrics={};ec={};st=time.monotonic()
    for arm in ARMS:
        status('evaluate',arm=arm);m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);ee,rr=m['E'].cuda(),m['R'].cuda();assert torch.equal(ee[:N].cpu(),initial['E'][:N].double()) and torch.equal(rr.cpu(),initial['R'].double());states={};cost={}
        for split in ['old_train','old_valid']:
            if fast_ok:states[split],cost[split]=fast_ranks(ee,rr,queries[split],zs[split],truths)
            else:states[split]=base.ranks(ee,rr,queries[split],truths);cost[split]=dict(fallback_all=True)
        s=time.monotonic();states['new_valid']=base.ranks(ee,rr,newvalid,truths);cost['new_valid']=dict(seconds=time.monotonic()-s);np.savez(out/f'{arm}.npz',**states)
        metric={split:base.metrics(states[split],q) for split,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],zs[split]['teacher'])
        metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS] if arm!='none' else [1.,1.,1.]
        metrics[arm]=metric;ec[arm]=cost;base.write(out/'metrics-running.json',metrics);print('EVALUATED112',arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],metric['directional_ratios'],flush=True)
    costs.update(evaluation=ec,evaluation_seconds=time.monotonic()-st,total_seconds=time.monotonic()-tick,fast_used=fast_ok)
    base.write(ROOT/'summary.json',dict(dataset='WN_CKGE',seed=SEED,metrics=metrics,costs=costs));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
