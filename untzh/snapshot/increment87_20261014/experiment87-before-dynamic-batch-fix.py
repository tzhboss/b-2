import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/blend_confirm80_20261014')
import experiment80 as core
base=core.base;ex=core.ex;ROOT=Path(__file__).resolve().parent
core.ROOT=ROOT;base.ROOT=ROOT
DATA=core.SRC/'literature_baselines_20261010/repos/FastKGE/data/FB_CKGE'
N,TOTAL=11258,13134
core.N=N;core.TOTAL=TOTAL;ex.N=N;ex.TOTAL=TOTAL;base.N=N;base.TOTAL=TOTAL

def mapping(ep,file):return {name:int(i) for name,i in (line.split() for line in (DATA/str(ep)/file).read_text().splitlines())}

def triples(ep,split):
    e=mapping(ep,'entity2id.txt');r=mapping(ep,'relation2id.txt');out=[]
    for line in (DATA/str(ep)/(split+'.txt')).read_text().splitlines():
        h,rr,g=line.split();h,rr,g=e[h],r[rr],e[g];assert rr%2==0
        out.extend([(h,rr,g),(g,rr+1,h)])
    return np.array(out,dtype=np.int64)

def prepare():
    tick=time.monotonic();(ROOT/'cache').mkdir()
    e1=mapping(1,'entity2id.txt');e2=mapping(2,'entity2id.txt');r1=mapping(1,'relation2id.txt');r2=mapping(2,'relation2id.txt')
    assert len(e1)==N and len(e2)==TOTAL and r1==r2 and all(e2[k]==v for k,v in e1.items())
    cached=base.load(core.SRC/'entity_diagnostic42_20261012/artifacts/data/train.npz')
    old=np.concatenate([cached['old'],cached['new']]);source=triples(2,'train')
    mask=(source[:,0]>=N)|(source[:,2]>=N);new=source[mask];extra=source[~mask]
    assert len(new)>0 and len(new)<=37214
    parent=torch.load(core.SRC/'matched_protection52_20261013/artifacts/train/new_only/epoch-20.pt',map_location='cpu',weights_only=True)
    assert parent['E'].shape==(N,200) and parent['R'].shape[0]==len(r1)
    gen=torch.Generator(device='cuda').manual_seed(1087);E=torch.empty((TOTAL,200),device='cuda',dtype=torch.float32)
    torch.nn.init.normal_(E,std=.12,generator=gen);E[:N]=parent['E'].cuda().float()
    # The parent was exported as double from FP32, so cast preserves old values.
    assert torch.equal(E[:N].cpu().double(),parent['E'])
    initial=dict(E=E.cpu(),R=parent['R'].float());torch.save(initial,ROOT/'cache/initial.pt')
    tr=core.truthmap(old,source)
    np.savez(ROOT/'cache/train.npz',old=old,new=new,extra=extra)
    touched=new[:,[0,2]];new_support=len(np.unique(touched[touched>=N]))
    report=dict(seconds=time.monotonic()-tick,old_train=len(old),source_new_train=len(source),new_train=len(new),extra_old_old=len(extra),old_entities=N,new_entities=TOTAL-N,relations=len(r1),new_rows_with_train_support=new_support)
    core.dump('data.json',report);core.progress('data',**report)
    return initial,old,new,tr,report

def evaluate(initial,old,new,tr,A):
    tick=time.monotonic();rng=np.random.default_rng(124)
    oldvalid=np.concatenate([triples(0,'valid'),triples(1,'valid')]);source=triples(2,'valid')
    mask=(source[:,0]>=N)|(source[:,2]>=N);newvalid=source[mask]
    oldvalid=oldvalid[rng.permutation(len(oldvalid))[:1024]];newvalid=newvalid[rng.permutation(len(newvalid))[:1024]]
    core.dump('validation-selection.json',dict(old_queries=len(oldvalid),new_queries=len(newvalid),new_source=len(source),new_novel_eligible=int(mask.sum()),seed=124))
    out=ROOT/'evaluation';out.mkdir();np.savez(out/'queries.npz',old_train=old,old_valid=oldvalid,new_valid=newvalid)
    av=base.ranks(initial['E'][:N].cuda().double(),initial['R'].cuda().double(),oldvalid,tr,arm='teacher',split='old_valid')
    np.savez(out/'teacher.npz',old_train=A,old_valid=av);report={}
    for arm in core.ARMS:
        m=torch.load(ROOT/'models'/arm/'epoch-20.pt',map_location='cpu',weights_only=True);E,R=m['E'].cuda(),m['R'].cuda()
        qs=dict(old_train=old,old_valid=oldvalid,new_valid=newvalid)
        states={k:base.ranks(E,R,q,tr,arm=arm,split=k) for k,q in qs.items()};np.savez(out/f'{arm}.npz',**states)
        report[arm]={k:base.metrics(states[k],q) for k,q in qs.items()}
        for k,ta in [('old_train',A),('old_valid',av)]:report[arm][k]['drops']=core.drops(states[k],ta)
        core.progress('evaluated',arm=arm,new_mrr=report[arm]['new_valid']['all']['filtered_train_truths']['mrr'],train_drops=report[arm]['old_train']['drops']['filtered_train_truths']['drops'])
    for arm,m in report.items():m['directional_ratios']=[m['new_valid'][d]['filtered_train_truths']['mrr']/report['none']['new_valid'][d]['filtered_train_truths']['mrr'] for d in core.DIRECTIONS]
    core.dump('evaluation/metrics.json',report);return report,time.monotonic()-tick

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.monotonic();core.progress('starting');initial,old,new,tr,dc=prepare()
    t,tc=core.tape(17,new,tr,len(initial['R']))
    # Number of batches depends on novel TRAIN count; this task expects 37 per epoch.
    assert len(t['epochs'])==740,('unexpected_batch_count',len(t['epochs']))
    A,ats=core.teacher(initial,old,tr)
    d,cc=core.cohort(initial,old,tr,A,17)
    p,ps=core.prefix10(initial,new,t);cal,cs=core.calibrate(initial,p,new,t,d)
    costs=dict(data=dc,tape=tc,teacher_seconds=ats,cohort=cc,prefix_seconds=ps,calibration_seconds=cs,training={})
    for arm in core.ARMS:costs['training'][arm]=core.train(arm,initial,p,new,t,d,cal);core.dump('costs-running.json',costs)
    report,es=evaluate(initial,old,new,tr,A);costs['evaluation_seconds']=es;costs['total_seconds']=time.monotonic()-tick
    core.dump('summary.json',dict(metrics=report,costs=costs));core.progress('complete',seconds=costs['total_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:core.progress('failed',error=repr(exc));raise
