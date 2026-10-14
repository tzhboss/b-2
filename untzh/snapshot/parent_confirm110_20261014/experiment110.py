import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import json
import collections
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/online_shadow106_20261014')
import experiment106 as online
sys.path.insert(0,'/data/lc/untzh/query_budget101_20261014')
import experiment101 as query
sys.path.insert(0,'/data/lc/untzh/increment87_20261014')
import experiment87 as existing
sys.path.insert(0,'/data/lc/untzh/fast_old_eval107_20261014')
import experiment107 as fastmodule
base,core,space_module=online.base,online.core,online.prior
ROOT=Path(__file__).resolve().parent
N,TOTAL,SEED=11258,13134,43
ARMS=['none','absolute','anchor_null','query_uniform','full_replay_joint','full_replay_anchor']

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def configure():
    online.configure(ROOT,ROOT);existing.ROOT=ROOT;query.ROOT=ROOT;space_module.ROOT=ROOT
    online.replay.ROOT=ROOT;online.replay.previous.ROOT=ROOT

@torch.no_grad()
def teacher_cache(initial,queries,truths,split):
    tick=time.monotonic();oldE=initial['E'][:N].cuda().double();R=initial['R'].cuda().double()
    ranks=[];golds=[];newrows=[];newcols=[]
    for lo in range(0,len(queries),64):
        b=queries[lo:lo+64];q=torch.as_tensor(b,device='cuda')
        scores=(oldE[q[:,0]]*R[q[:,1]])@oldE.T
        gold=scores[torch.arange(len(b),device='cuda'),q[:,2]].clone()
        raw=1+(scores>gold[:,None]).sum(1)
        rows=[];cols=[]
        for j,(h,r,g) in enumerate(b):
            for c in truths[int(h),int(r)]:
                if c>=N:newrows.append(lo+j);newcols.append(c-N)
                elif c!=int(g):rows.append(j);cols.append(c)
        if rows:scores[rows,cols]=-torch.inf
        filtered=1+(scores>gold[:,None]).sum(1)
        ranks.append(torch.stack([raw,filtered],1).cpu().numpy());golds.append(gold.cpu().numpy())
    payload=dict(gold=np.concatenate(golds),rows=np.array(newrows,dtype=np.int64),cols=np.array(newcols,dtype=np.int64))
    teacher=np.concatenate(ranks);np.savez(ROOT/'cache'/f'fast-{split}.npz',teacher=teacher,**payload)
    return teacher,payload,dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in payload.values())+teacher.nbytes,eligible_filtered=int((teacher[:,1]<=10).sum()))

def full_replay(initial,old,truths):
    tick=time.monotonic();rng=np.random.default_rng(1040000+SEED)
    pos=rng.integers(len(old),size=440*32,dtype=np.int32);p=old[pos[220*32:]]
    h,r,t=p.T;e=initial['E'].double();rel=initial['R'].double()
    qt=(e[h]*rel[r]).numpy().reshape(220,32,200);qh=(e[t]*rel[r]).numpy().reshape(220,32,200)
    heads=collections.defaultdict(set)
    for (hh,rr),ts in truths.items():
        for tt in ts:heads[rr,tt].add(hh)
    mt=np.ones((len(p),TOTAL-N),bool);mh=np.ones_like(mt);dt=np.empty(len(p));dh=np.empty(len(p))
    for i,(hh,rr,tt) in enumerate(p):
        tails=truths[int(hh),int(rr)];hs=heads[int(rr),int(tt)]
        mt[i,[x-N for x in tails if x>=N]]=False;mh[i,[x-N for x in hs if x>=N]]=False
        dt[i]=TOTAL-len(tails);dh[i]=TOTAL-len(hs)
    payload=dict(qtail=qt,qhead=qh,mask_tail=mt.reshape(220,32,-1),mask_head=mh.reshape(220,32,-1),denominator_tail=dt.reshape(220,32),denominator_head=dh.reshape(220,32),positions=pos)
    np.savez(ROOT/'cache/full-replay.npz',**payload)
    return {k:torch.as_tensor(v,device='cuda') for k,v in payload.items() if k!='positions'},dict(seconds=time.monotonic()-tick,bytes=sum(x.nbytes for x in payload.values()),sampled_once=True)

def main():
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();configure();status('prepare');(ROOT/'cache').mkdir()
    base.write(ROOT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=SEED))
    start=time.monotonic();z=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')
    old,new,extra=z['old'],z['new'],z['extra'];truths=core.truthmap(old,np.concatenate([new,extra]));np.savez(ROOT/'cache/train.npz',**z)
    parentpath=Path('/data/lc/untzh/blend_confirm80_20261014/seed43/models/none/epoch-20.pt')
    parent=torch.load(parentpath,map_location='cpu',weights_only=True)
    assert parent['E'].shape==(N,200) and parent['R'].shape==(474,200)
    E=torch.empty((TOTAL,200),device='cuda',dtype=torch.float32)
    torch.nn.init.normal_(E,std=.12,generator=torch.Generator(device='cuda').manual_seed(1070+SEED));E[:N]=parent['E'].cuda().float()
    assert torch.equal(E[:N].cpu().double(),parent['E'])
    initial=dict(E=E.cpu(),R=parent['R'].float());torch.save(initial,ROOT/'cache/initial.pt')
    costs=dict(initial_seconds=time.monotonic()-start,parent=str(parentpath),historical_assets_not_free=True)
    tape,costs['learning_tape']=core.tape(SEED,new,truths,len(initial['R']));assert len(tape['epochs'])==440
    status('teacher_train');A,oldcache,costs['teacher_train']=teacher_cache(initial,old,truths,'old_train')
    np.savez(ROOT/'cache/teacher-train.npz',old_train=A)
    data,costs['cohort']=core.cohort(initial,old,truths,A,SEED)
    prefix,costs['prefix_seconds'],first=existing.prefix10(initial,new,tape);assert first==220
    calibration,costs['calibration_seconds']=existing.calibrate(initial,prefix,new,tape,data,220)
    c=base.load(ROOT/'cache/cohort.npz');ranks=A[c['ids'],1]
    data['fixed']=np.column_stack([ranks,np.zeros((len(ranks),7))])
    data['gold']=(data['q']*initial['E'].double().cuda()[c['queries'][:,2]]).sum(-1)
    online.action.extra_data(truths,data)
    space,costs['space']=space_module.build_space(initial,new)
    replaydata,costs['full_replay_cache']=full_replay(initial,old,truths)
    groups,representatives,inverse=np.unique(c['queries'][:,:2],axis=0,return_index=True,return_inverse=True)
    assert np.array_equal(c['q'],c['q'][representatives[inverse]])
    np.savez(ROOT/'cache/query-groups.npz',groups=groups,representatives=representatives,inverse=inverse)
    costs['query_groups']=len(groups);costs['preparation_seconds']=time.monotonic()-tick;costs['training']={}
    for arm in ARMS:
        status('train',arm=arm)
        if arm in ['none','absolute']:
            report=existing.train(arm,initial,prefix,new,tape,data,calibration,220)
        else:
            setup=time.monotonic();anchor=online.OnlineAnchor(initial,prefix,new,tape);setup_seconds=time.monotonic()-setup
            if arm=='anchor_null':report=online.absolute.train(arm,initial,prefix,new,tape,data,space,anchor)
            elif arm=='query_uniform':report=query.train(arm,initial,prefix,new,tape,data,None,space,anchor,representatives,inverse)
            else:report=online.replay.train(arm,initial,prefix,new,tape,replaydata,space,anchor,ROOT)
            ordinary=torch.load(ROOT/'models/none/epoch-20.pt',map_location='cpu',weights_only=True)
            assert torch.equal(anchor.E.detach().cpu().double(),ordinary['E']) and torch.equal(anchor.R.cpu().double(),ordinary['R'])
            report.update(online_setup_seconds=setup_seconds,shadow_compute_included=True,ordinary_endpoint_equal=True)
            del anchor
        costs['training'][arm]=report;base.write(ROOT/'costs-running.json',costs)
    # All six models committed before first VALID read.
    status('evaluation_prepare');full=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')
    av,validcache,costs['teacher_valid']=teacher_cache(initial,full['old_valid'],truths,'old_valid')
    fast=fastmodule.FastOld.__new__(fastmodule.FastOld)
    fast.seed=SEED;fast.truths=truths;fast.oldE=initial['E'][:N].cuda().double();fast.R=initial['R'].cuda().double()
    fast.queries=dict(old_train=old,old_valid=full['old_valid']);fast.teacher=dict(old_train=A,old_valid=av)
    fast.cache={s:dict(gold=torch.as_tensor(d['gold'],device='cuda'),rows=d['rows'],cols=d['cols']) for s,d in [('old_train',oldcache),('old_valid',validcache)]}
    queries=dict(fast.queries,new_valid=full['new_valid']);checks=[];fast_ok=True
    for arm in ['none','query_uniform']:
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);ee,rr=m['E'].cuda(),m['R'].cuda()
        for split in ['old_train','old_valid']:
            state,_=fast.ranks(ee,rr,split)
            starts=[0,(len(queries[split])//2//64)*64,((len(queries[split])-64)//64)*64]
            for lo in starts:
                exact=base.ranks(ee,rr,queries[split][lo:lo+64],truths)
                equal=np.array_equal(exact,state[lo:lo+64]);fast_ok&=equal;checks.append(dict(arm=arm,split=split,start=lo,n=64,equal=equal))
    base.write(ROOT/'fast-spot-checks.json',dict(all_equal=fast_ok,checks=checks,scope='fixed samples, not full-rank re-audit; previous107 full arrays validated implementation'))
    evaluation=ROOT/'evaluation';evaluation.mkdir();np.savez(evaluation/'queries.npz',**queries);np.savez(evaluation/'teacher.npz',old_train=A,old_valid=av)
    metrics={};ec={};start=time.monotonic()
    for arm in ARMS:
        status('evaluate',arm=arm);m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True)
        ee,rr=m['E'].cuda(),m['R'].cuda();states={};cost={}
        for split in ['old_train','old_valid']:
            if fast_ok:states[split],cost[split]=fast.ranks(ee,rr,split)
            else:
                t=time.monotonic();states[split]=base.ranks(ee,rr,queries[split],truths,arm=arm,split=split);cost[split]=dict(seconds=time.monotonic()-t,fallback_all=True)
        t=time.monotonic();states['new_valid']=base.ranks(ee,rr,queries['new_valid'],truths,arm=arm,split='new_valid');cost['new_valid']=dict(seconds=time.monotonic()-t)
        np.savez(evaluation/f'{arm}.npz',**states)
        metric={s:base.metrics(states[s],q) for s,q in queries.items()}
        for split in ['old_train','old_valid']:metric[split]['drops']=core.drops(states[split],fast.teacher[split])
        metric['directional_ratios']=[metric['new_valid'][d]['filtered_train_truths']['mrr']/metrics['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS] if arm!='none' else [1.,1.,1.]
        metrics[arm]=metric;ec[arm]=cost;base.write(evaluation/'metrics-running.json',metrics)
        print('EVALUATED110',arm,metric['new_valid']['all']['filtered_train_truths']['mrr'],metric['old_valid']['drops']['filtered_train_truths']['drops'],metric['directional_ratios'],flush=True)
    costs.update(evaluation=ec,evaluation_seconds=time.monotonic()-start,total_seconds=time.monotonic()-tick,fast_used=fast_ok)
    base.write(ROOT/'summary.json',dict(seed=SEED,metrics=metrics,costs=costs));status('complete',seconds=time.monotonic()-tick)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
