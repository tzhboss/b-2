"""Standalone independent inference/aggregation audit for embedding exports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);a=p.parse_args()
root=a.root.resolve();sys.path.insert(0,str(root/'src'))
from common_training import sha
torch.set_num_threads(2)
cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
out=root/'artifacts/incremental'/a.job;checks=0;failures=[]
def check(ok,label):
    global checks
    checks+=1
    if not ok:failures.append(label)
t=json.loads((out/'training-receipt.json').read_text());e=json.loads((out/'evaluation-receipt.json').read_text())
check(not t['test_read'],'No TEST during fitting');check(sha(out/'best.pt')==t['checkpoint_sha256'],'Checkpoint hash')
for path,digest in dict(t['bound_input_sha256'],**e['test_input_sha256']).items():check(sha(path)==digest,'Input hash '+path)
for name,digest in e['output_sha256'].items():check(sha(out/name)==digest,'Output hash '+name)
c=torch.load(out/'best.pt',map_location='cpu',weights_only=True);E=c['E'].numpy();R=c['R'].numpy()
check(E.shape==(11258,job['after_dimension']) and R.shape==(474,job['after_dimension']),'Dimension and candidate contract')
check(np.isfinite(E).all() and np.isfinite(R).all(),'Finite effective model')
f=np.load(out/'audit-fixture.npz');q=f['queries'];saved=f['scores'];kind=job['kind']
if kind=='TransE':
    En=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-12);Rn=R/np.maximum(np.linalg.norm(R,axis=1,keepdims=True),1e-12)
    actual=np.stack([1/(1+np.exp(-(9-np.abs(En[h]+Rn[r]-En).sum(1)))) for h,r,t in q]);tol=1e-12
else:
    actual=np.stack([(E[h]*R[r])@E.T for h,r,t in q]);tol=2e-4
error=float(np.abs(actual-saved).max());check(error<=tol,'64 full candidate score rows independently replayed')
actual_rank=1+(actual>actual[np.arange(len(q)),q[:,2]][:,None]).sum(1)
saved_rank=1+(saved>saved[np.arange(len(q)),q[:,2]][:,None]).sum(1)
check(np.array_equal(actual_rank,saved_rank),'Exact independently replayed ranks')
table=np.load(out/'ranks.npz');summary=json.loads((out/'evaluation.json').read_text())
check(np.array_equal(q,np.concatenate([table['old_queries'][:32],table['new_queries'][:32]])),'Fixture query identity')
check(np.array_equal(actual_rank,np.concatenate([table['old_after'][:32],table['new_after'][:32]])),'Full export vs inference fixture')
source=(root/cfg['source_root']/cfg['dataset']).resolve()
for s,label in [(0,'old'),(1,'new')]:
    folder=source/str(s)
    entity=dict((k,int(v)) for k,v in (x.split() for x in (folder/'entity2id.txt').read_text().splitlines()))
    relation=dict((k,int(v)) for k,v in (x.split() for x in (folder/'relation2id.txt').read_text().splitlines()))
    expected=[]
    for line in (folder/'test.txt').read_text().splitlines():
        h,r,t=line.split();h,r,t=entity[h],relation[r],entity[t];expected.extend([(h,r,t),(t,r+1,h)])
    check(np.array_equal(table[label+'_queries'],expected),'Complete original TEST query IDs '+label)
for label in ['old_before','old_after','new_after']:
    rank=table[label];check(((rank>=1)&(rank<=(7505 if label=='old_before' else 11258))).all(),'Rank bounds '+label)
    for key,value in [('raw_mrr',(1/rank).mean()),('hits1',(rank<=1).mean()),('hits10',(rank<=10).mean())]:
        check(abs(summary[label][key]-value)<1e-14,'Full rank aggregate '+label+' '+key)
for before,key in [('old_before','old_top10_loss'),('old_expanded','same_pool_top10_loss')]:
    good=table[before]<=10;check(good.any() and abs(summary[key]-(table['old_after'][good]>10).mean())<1e-14,key)
check(abs(summary['mean_rank_displacement']-(table['old_after']-table['old_before']).mean())<1e-12,'Mean displacement')
if job['method']=='SAGE':
    mapped=np.load(out/'mapping-train.npz')['triples'];check(((mapped[:,[0,2]]>=0)&(mapped[:,[0,2]]<7505)).all(),'Mapping learned only on old entities')
    trainset=set()
    folder=source/'0';entity=dict((k,int(v)) for k,v in (x.split() for x in (folder/'entity2id.txt').read_text().splitlines()))
    relation=dict((k,int(v)) for k,v in (x.split() for x in (folder/'relation2id.txt').read_text().splitlines()))
    for line in (folder/'train.txt').read_text().splitlines():
        h,r,t=line.split();h,r,t=entity[h],relation[r],entity[t];trainset.update([(h,r,t),(t,r+1,h)])
    check(all(tuple(map(int,row)) in trainset for row in mapped),'Mapping supports subset of old TRAIN')
result=dict(status='PASS_INFERENCE_AND_AGGREGATION' if not failures else 'FAIL_RETAINED',checks=checks,failures=failures,
            candidate_score_checks=int(actual.size),max_abs_score_error=error,fixture_rank_mismatches=int((actual_rank!=saved_rank).sum()),
            auditor_sha256=sha(Path(__file__)),limits=['64 full candidate rows replayed; no independent optimizer retraining or whole-table score reconstruction.'])
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));sys.exit(bool(failures))
