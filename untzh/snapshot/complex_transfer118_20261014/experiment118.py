import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import copy
import collections
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/data/lc/untzh/wn_transfer112_20261014')
import experiment112 as source
base,core=source.base,source.core
ROOT=Path(__file__).resolve().parent
SRC=Path('/data/lc/untzh/wn_transfer112_20261014')
N,TOTAL,D=24567,28660,200
ARMS=['standard_none','none','absolute','answer_anchor','query_anchor','full_replay_joint','full_replay_anchor']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))
def tail_query(h,r):
    a,b=h.chunk(2,-1);c,d=r.chunk(2,-1)
    return torch.cat([a*c-b*d,a*d+b*c],-1)
def head_query(t,r):
    u,v=t.chunk(2,-1);c,d=r.chunk(2,-1)
    return torch.cat([u*c+v*d,v*c-u*d],-1)
def score(h,r,t):return (tail_query(h,r)*t).sum(-1)

def learning(E,R,facts,tape,j):
    lo,hi=tape['bounds'][j:j+2];p=torch.as_tensor(facts[tape['pos'][lo:hi]],device='cuda')
    nh=torch.as_tensor(tape['nh'][lo:hi],device='cuda',dtype=torch.long);nt=torch.as_tensor(tape['nt'][lo:hi],device='cuda',dtype=torch.long)
    h,r,t=p.unbind(1);positive=score(E[h],R[r],E[t]);negative=score(E[nh],R[r[:,None]],E[nt])
    return F.softplus(-positive).mean()+F.softplus(negative).mean()+.0005*(E[h].square().mean()+R[r].square().mean())

def verify_formula(parent,facts):
    p=facts[:64];h=parent['E'][p[:,0]].double().requires_grad_();t=parent['E'][p[:,2]].double().requires_grad_();r=parent['R'][p[:,1]].double()
    a,b=h.chunk(2,-1);c,d=r.chunk(2,-1);u,v=t.chunk(2,-1)
    explicit=(a*c*u+a*d*v+b*c*v-b*d*u).sum(-1);s=score(h,r,t);gh,gt=torch.autograd.grad(s.sum(),[h,t])
    report=dict(explicit_score_max_error=float((explicit-s).abs().max()),head_form_max_error=float((s-(head_query(t,r)*h).sum(-1)).abs().max()),head_gradient_max_error=float((gh-head_query(t,r)).abs().max()),tail_gradient_max_error=float((gt-tail_query(h,r)).abs().max()),trained_relation_imag_max=float(parent['R'][:,100:].abs().max()),swapped_score_max_difference=float((s-score(t,r,h)).abs().max()))
    assert max(report[k] for k in ['explicit_score_max_error','head_form_max_error','head_gradient_max_error','tail_gradient_max_error'])<1e-10
    assert report['trained_relation_imag_max']>1e-8 and report['swapped_score_max_difference']>1e-8
    base.write(ROOT/'formula-check.json',report)

@torch.no_grad()
def ranks(E,R,queries,truths):
    result=[]
    for lo in range(0,len(queries),64):
        b=queries[lo:lo+64];a=torch.as_tensor(b,device='cuda');ss=tail_query(E[a[:,0]],R[a[:,1]])@E.T;gold=ss[torch.arange(len(b),device='cuda'),a[:,2]].clone()
        raw=1+(ss>gold[:,None]).sum(1);rows=[];cols=[]
        for j,(h,r,t) in enumerate(b):
            for c in truths[int(h),int(r)]:
                if c!=int(t) and c<len(E):rows.append(j);cols.append(c)
        if rows:ss[rows,cols]=-torch.inf
        result.append(torch.stack([raw,1+(ss>gold[:,None]).sum(1)],1).cpu().numpy())
    return np.concatenate(result)

@torch.no_grad()
def teacher_cache(initial,queries,truths,split):
    tick=time.monotonic();E=initial['E'][:N].cuda().double();R=initial['R'].cuda().double();parts=[];golds=[];nr=[];nc=[]
    for lo in range(0,len(queries),64):
        b=queries[lo:lo+64];a=torch.as_tensor(b,device='cuda');ss=tail_query(E[a[:,0]],R[a[:,1]])@E.T;gold=ss[torch.arange(len(b),device='cuda'),a[:,2]].clone();raw=1+(ss>gold[:,None]).sum(1);rows=[];cols=[]
        for j,(h,r,t) in enumerate(b):
            for c in truths[int(h),int(r)]:
                if c>=N:nr.append(lo+j);nc.append(c-N)
                elif c!=int(t):rows.append(j);cols.append(c)
        if rows:ss[rows,cols]=-torch.inf
        parts.append(torch.stack([raw,1+(ss>gold[:,None]).sum(1)],1).cpu().numpy());golds.append(gold.cpu().numpy())
    z=dict(teacher=np.concatenate(parts),gold=np.concatenate(golds),rows=np.array(nr,dtype=np.int64),cols=np.array(nc,dtype=np.int64));np.savez(ROOT/'cache'/f'fast-{split}.npz',**z)
    return z,dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in z.values()),eligible=int((z['teacher'][:,1]<=10).sum()))

@torch.no_grad()
def fast_ranks(E,R,queries,z,truths):
    tick=time.monotonic();parts=[];fallback=0
    for lo in range(0,len(queries),64):
        b=queries[lo:lo+64];a=torch.as_tensor(b,device='cuda');gold=torch.as_tensor(z['gold'][lo:lo+len(b)],device='cuda');ss=tail_query(E[a[:,0]],R[a[:,1]])@E[N:].T
        if ((ss-gold[:,None]).abs()<=1e-10*(1+gold.abs()[:,None])).any():parts.append(ranks(E,R,b,truths));fallback+=len(b);continue
        raw=(ss>gold[:,None]).sum(1).cpu().numpy();i=np.searchsorted(z['rows'],lo);j=np.searchsorted(z['rows'],lo+len(b))
        if j>i:ss[z['rows'][i:j]-lo,z['cols'][i:j]]=-torch.inf
        filtered=(ss>gold[:,None]).sum(1).cpu().numpy();parts.append(z['teacher'][lo:lo+len(b)]+np.column_stack([raw,filtered]))
    return np.concatenate(parts),dict(seconds=time.monotonic()-tick,fallback_queries=fallback)

def build_space(initial,new):
    tick=time.monotonic();keys=[set() for _ in range(TOTAL-N)];cross=0
    for h,r,t in new:
        if h<N<=t:keys[int(t)-N].add((int(h),int(r),0));cross+=1
        elif t<N<=h:keys[int(h)-N].add((int(t),int(r),1));cross+=1
    E,R=initial['E'].double(),initial['R'].double();packed=collections.defaultdict(list);rows=[];vectors=[];dimensions=[]
    for i,k in enumerate(keys):
        if not k:dimensions.append(0);continue
        a=np.array(sorted(k));neighbor,rel,side=a.T;qt=tail_query(E[neighbor],R[rel]);qh=head_query(E[neighbor],R[rel]);q=torch.where(torch.as_tensor(side[:,None])==0,qt,qh)
        rows.extend([i]*len(q));vectors.append(q);norm=q.norm(dim=-1);q=q[norm>0]/norm[norm>0,None]
        if len(q):_,s,vh=torch.linalg.svd(q,full_matrices=False);rank=int((s>1e-10*s[0]).sum())
        else:rank=0
        dimensions.append(rank)
        if rank:
            width=min(D,2**int(np.ceil(np.log2(rank))));basis=torch.zeros((D,width),dtype=torch.float64);basis[:,:rank]=vh[:rank].T;packed[width].append((i,basis))
    groups=[dict(indices=torch.tensor([i for i,b in items]),basis=torch.stack([b for i,b in items])) for items in packed.values()]
    payload=dict(groups=groups,rows=torch.tensor(rows),vectors=torch.cat(vectors),ranks=dimensions);torch.save(payload,ROOT/'cache/query-space.pt')
    report=dict(seconds=time.monotonic()-tick,cross_facts=cross,unique_constraints=len(rows),mean_rank=float(np.mean(dimensions)),max_rank=max(dimensions),zero_rank=dimensions.count(0),basis_bytes=sum(g['basis'].numel()*8 for g in groups),head_tail_roles_separate=True)
    base.write(ROOT/'space.json',report)
    for g in groups:g['indices']=g['indices'].cuda();g['basis']=g['basis'].cuda()
    payload['rows']=payload['rows'].cuda();payload['vectors']=payload['vectors'].cuda();return payload,report

def protection(E,data,chosen,full):
    if not full:return F.softplus((data['q'][chosen,None]*E[data['candidates'][chosen]].double()).sum(-1)).mean()
    ss=data['q'][chosen]@E[N:].double().T;legal=data['legal'][chosen]
    return (F.softplus(ss).masked_fill(~legal,0).sum(-1)/legal.sum(-1)).mean()

def replay_penalty(E,data,k):
    tail=F.softplus(data['qt'][k]@E[N:].double().T).masked_fill(~data['mt'][k],0).sum(-1)/data['dt'][k]
    head=F.softplus(data['qh'][k]@E[N:].double().T).masked_fill(~data['mh'][k],0).sum(-1)/data['dh'][k]
    return .5*(tail+head).mean()

def prefix10(initial,new,tape,start):
    tick=time.monotonic();E,R,opt=source.model(initial)
    for j in range(start):loss=learning(E,R,new,tape,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(E).all()
    return dict(E=E.detach().cpu(),R=R.cpu(),optimizer=copy.deepcopy(opt.state_dict())),time.monotonic()-tick

def train(arm,initial,prefix,new,tape,data,space,replay,calibration,start):
    E,R,opt=source.model(initial,prefix);shadow=None if arm in ['none','standard_none','absolute'] else source.model(initial,prefix);rng=np.random.default_rng(1010029);records=[];tick=time.monotonic()
    for j in range(start,len(tape['epochs'])):
        k=j-start;loss=learning(E,R,new,tape,j);gl=torch.autograd.grad(loss,E)[0];bn=float(gl[N:].double().norm());pn=0.;alpha=0.
        if arm in ['none','standard_none']:gradient=gl
        else:
            if arm.startswith('full_replay'):p=replay_penalty(E,replay,k)
            else:
                chosen=data['representatives'][rng.choice(len(data['representatives']),32,replace=False)] if arm=='query_anchor' else data['selections'][k]
                p=protection(E,data,chosen,arm!='absolute')
            gp=torch.autograd.grad(p,E)[0];pn=float(gp[N:].double().norm());alpha=min(calibration,.1*bn/pn) if arm=='absolute' and pn else .1*bn/pn if pn else 0.;gradient=gl+alpha*gp
        opt.zero_grad(set_to_none=True);E.grad=gradient;opt.step();cross=None;norm=None
        if shadow:
            se,sr,so=shadow;sl=learning(se,sr,new,tape,j);so.zero_grad(set_to_none=True);sl.backward();so.step();reference=se.detach()[N:]
            difference=(E.detach()[N:]-reference).double()
            if arm!='full_replay_joint':
                accepted=source.space_module.project(difference,space)
                with torch.no_grad():E[N:].copy_(reference+accepted.to(E.dtype))
            actual=(E.detach()[N:]-reference).double();cross=float((space['vectors']*actual[space['rows']]).sum(-1).abs().max());norm=float(actual.norm())
        assert torch.isfinite(E).all();records.append(dict(step=j,gradient_ratio=alpha*pn/bn if bn else 0.,cross_max=cross,norm=norm))
    assert torch.equal(E[:N].cpu(),initial['E'][:N]) and torch.equal(R.cpu(),initial['R'])
    if shadow:
        ordinary=torch.load(ROOT/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
        assert torch.equal(shadow[0].detach().cpu().double(),ordinary['E']) and torch.equal(shadow[1].cpu().double(),ordinary['R'])
    out=ROOT/'models'/arm;out.mkdir(parents=True);torch.save(dict(E=E.detach().cpu().double(),R=R.cpu().double()),out/'epoch-20.pt');base.write(out/'steps.json',records)
    report=dict(seconds=time.monotonic()-tick,steps=len(records),mean_gradient_ratio=float(np.mean([x['gradient_ratio'] for x in records])),max_cross_score_difference=max((x['cross_max'] for x in records if x['cross_max'] is not None),default=None),final_norm=records[-1]['norm'],frozen_equal=True,shadow_compute_included=bool(shadow),ordinary_endpoint_equal=True if shadow else None)
    base.write(out/'training.json',report);return report

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.monotonic();base.ROOT=core.ROOT=ROOT;base.N=core.N=N;base.TOTAL=core.TOTAL=TOTAL;status('prepare');(ROOT/'cache').mkdir()
    base.write(ROOT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),model='ComplEx100,200 real parameters'))
    z=base.load(SRC/'cache/train.npz');old,new=z['old'],z['new'];truths=core.truthmap(old,new);costs={}
    t=time.monotonic();pt=base.load(SRC/'cache/parent-learning.npz');tape=base.load(SRC/'cache/learning.npz');costs['tape_read_seconds']=time.monotonic()-t;costs['tapes_reused_without_resampling']=True
    E=torch.nn.Parameter(torch.randn((N,D),device='cuda')*.12);R=torch.nn.Parameter(torch.randn((22,D),device='cuda')*.12);opt=torch.optim.Adam([E,R],lr=.007);t=time.monotonic()
    for j in range(len(pt['epochs'])):
        loss=learning(E,R,old,pt,j);opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(E).all() and torch.isfinite(R).all()
        if j==len(pt['epochs'])-1 or pt['epochs'][j+1]!=pt['epochs'][j]:status('parent_train',epoch=int(pt['epochs'][j]))
    costs['parent_training_seconds']=time.monotonic()-t;parent=dict(E=E.detach().cpu(),R=R.detach().cpu());torch.save(parent,ROOT/'cache/parent.pt');verify_formula(parent,old);del pt,E,R,opt
    expanded=torch.empty((TOTAL,D),device='cuda');torch.nn.init.normal_(expanded,std=.12,generator=torch.Generator(device='cuda').manual_seed(1087));expanded[:N]=parent['E'].cuda()
    standard=dict(E=expanded.cpu(),R=parent['R']);initial=copy.deepcopy(standard);initial['E'][N:]*=.25;torch.save(standard,ROOT/'cache/standard-initial.pt');torch.save(initial,ROOT/'cache/initial.pt')
    start=int(np.flatnonzero(tape['epochs']==11)[0]);prefix,costs['low_prefix_seconds']=prefix10(initial,new,tape,start);standard_prefix,costs['standard_prefix_seconds']=prefix10(standard,new,tape,start)
    torch.save(prefix,ROOT/'cache/prefix10.pt');torch.save(standard_prefix,ROOT/'cache/standard-prefix10.pt')
    status('teacher_train');traincache,costs['teacher_train']=teacher_cache(initial,old,truths,'old_train')
    t=time.monotonic();rng=np.random.default_rng(17);eligible=np.flatnonzero(traincache['teacher'][:,1]<=10);size=min(16384,len(eligible));assert size>=32
    ids=rng.choice(eligible,size,replace=False);queries=old[ids];candidates=rng.integers(N,TOTAL,size=(size,128));legal=np.ones((size,TOTAL-N),bool)
    for i,(h,r,g) in enumerate(queries):
        known=list(truths[int(h),int(r)]);bad=np.isin(candidates[i],known)
        while bad.any():candidates[i,bad]=rng.integers(N,TOTAL,size=int(bad.sum()));bad=np.isin(candidates[i],known)
        legal[i,[c-N for c in known if c>=N]]=False
    q=tail_query(initial['E'][queries[:,0]].double(),initial['R'][queries[:,1]].double());groups,reps,inverse=np.unique(queries[:,:2],axis=0,return_index=True,return_inverse=True)
    selections=np.stack([np.random.default_rng(17+1000+j).choice(size,32,replace=False) for j in range(len(tape['epochs'])-start)])
    np.savez(ROOT/'cache/cohort.npz',ids=ids,queries=queries,q=q.numpy(),candidates=candidates,legal=legal,representatives=reps,inverse=inverse,selections=selections)
    data=dict(q=q.cuda(),candidates=torch.as_tensor(candidates,device='cuda'),legal=torch.as_tensor(legal,device='cuda'),representatives=reps,selections=selections);costs['cohort']=dict(seconds=time.monotonic()-t,eligible=len(eligible),size=size,query_groups=len(groups))
    space,costs['space']=build_space(initial,new)
    t=time.monotonic();E,R,opt=source.model(initial,prefix);norms=[]
    for k in range(8):
        gl=torch.autograd.grad(learning(E,R,new,tape,start+k),E)[0];gp=torch.autograd.grad(protection(E,data,selections[k],False),E)[0];norms.append([float(gl[N:].double().norm()),float(gp[N:].double().norm())])
    a=np.array(norms);calibration=.1*np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2));costs['calibration_seconds']=time.monotonic()-t;base.write(ROOT/'calibration.json',dict(coefficient=calibration,norms=norms))
    t=time.monotonic();replayz=base.load(SRC/'cache/full-replay.npz');p=old[replayz['positions'][start*32:]];count=len(tape['epochs'])-start
    replayz['qt']=tail_query(initial['E'][p[:,0]].double(),initial['R'][p[:,1]].double()).numpy().reshape(count,32,D)
    replayz['qh']=head_query(initial['E'][p[:,2]].double(),initial['R'][p[:,1]].double()).numpy().reshape(count,32,D)
    np.savez(ROOT/'cache/full-replay.npz',**replayz);replay={k:torch.as_tensor(v,device='cuda') for k,v in replayz.items() if k!='positions'};costs['replay_cache']=dict(seconds=time.monotonic()-t,bytes=sum(v.nbytes for v in replayz.values()),masks_positives_denominators_reused=True)
    costs.update(preparation_seconds=time.monotonic()-tick,historical_assets_not_free=True,training={})
    for arm in ARMS:
        status('train',arm=arm);costs['training'][arm]=train(arm,standard if arm=='standard_none' else initial,standard_prefix if arm=='standard_none' else prefix,new,tape,data,space,replay,calibration,start);base.write(ROOT/'costs-running.json',costs)
    # All seven models committed before any current VALID read.
    status('evaluation_prepare');saved=base.load(SRC/'evaluation/queries.npz');queries=dict(old_train=old,old_valid=saved['old_valid'],new_valid=saved['new_valid'])
    validcache,costs['teacher_valid']=teacher_cache(initial,queries['old_valid'],truths,'old_valid');zs=dict(old_train=traincache,old_valid=validcache)
    checks=[];fast_ok=True
    for arm in ['none','query_anchor']:
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda()
        for split in ['old_train','old_valid']:
            fast,_=fast_ranks(E,R,queries[split],zs[split],truths)
            for lo in [0,(len(queries[split])//2//64)*64,((len(queries[split])-64)//64)*64]:
                exact=ranks(E,R,queries[split][lo:lo+64],truths);equal=np.array_equal(fast[lo:lo+64],exact);fast_ok&=equal;checks.append(dict(arm=arm,split=split,start=lo,equal=equal))
    base.write(ROOT/'fast-spot-checks.json',dict(all_equal=fast_ok,checks=checks))
    out=ROOT/'evaluation';out.mkdir();np.savez(out/'queries.npz',**queries);np.savez(out/'teacher.npz',**{s:z['teacher'] for s,z in zs.items()});metrics={};ec={};t=time.monotonic()
    for arm in ARMS:
        status('evaluate',arm=arm);m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda();assert torch.equal(E[:N].cpu(),initial['E'][:N].double()) and torch.equal(R.cpu(),initial['R'].double());states={};cost={}
        for split in ['old_train','old_valid']:
            if fast_ok:states[split],cost[split]=fast_ranks(E,R,queries[split],zs[split],truths)
            else:states[split]=ranks(E,R,queries[split],truths);cost[split]=dict(fallback_all=True)
        s=time.monotonic();states['new_valid']=ranks(E,R,queries['new_valid'],truths);cost['new_valid']=dict(seconds=time.monotonic()-s);np.savez(out/f'{arm}.npz',**states)
        metric={split:base.metrics(states[split],q) for split,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],zs[split]['teacher'])
        metrics[arm]=metric;ec[arm]=cost;base.write(out/'metrics-running.json',metrics);print('EVALUATED118',arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],flush=True)
    for arm,m in metrics.items():m['directional_ratios_to_low_none']=[m['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
    stages={}
    for mode,init,pref,endpoint in [('standard',standard,standard_prefix,'standard_none'),('low',initial,prefix,'none')]:
        stages[mode]={}
        for split in ['old_train','old_valid']:
            a,c0=fast_ranks(init['E'].cuda().double(),init['R'].cuda().double(),queries[split],zs[split],truths);b,c10=fast_ranks(pref['E'].cuda().double(),init['R'].cuda().double(),queries[split],zs[split],truths)
            np.savez(out/f'{mode}-{split}-phases.npz',old_candidates=zs[split]['teacher'],expanded_initial=a,epoch10=b,epoch20=base.load(out/f'{endpoint}.npz')[split]);stages[mode][split]=dict(expanded=c0,epoch10=c10)
    costs.update(evaluation=ec,phase_scoring=stages,evaluation_seconds=time.monotonic()-t,total_seconds=time.monotonic()-tick,fast_used=fast_ok)
    base.write(ROOT/'summary.json',dict(model='ComplEx100',dataset='WN_CKGE',seed=17,metrics=metrics,costs=costs));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
