"""Independent NumPy inference replay and raw-rank aggregation audit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
root=a.root.resolve();sys.path.insert(0,str(root/'src'))
from common_training import sha
cfg=json.loads((root/'configs/incremental-training.json').read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
out=root/'artifacts/incremental'/a.job; failures=[]; checks=0
def check(value,message):
    global checks
    checks+=1
    if not value:failures.append(message)
receipt=json.loads((out/'training-receipt.json').read_text())
check(not receipt['test_read'],'Fitting must not read TEST')
check(sha(out/'best.pt')==receipt['checkpoint_sha256'],'Checkpoint hash')
for path,digest in receipt['bound_input_sha256'].items():check(sha(path)==digest,'Fitting input hash '+path)
evalreceipt=json.loads((out/'evaluation-receipt.json').read_text())
for name,digest in evalreceipt['output_sha256'].items():check(sha(out/name)==digest,'Evaluation output hash '+name)
for path,digest in evalreceipt['test_input_sha256'].items():check(sha(path)==digest,'Test input hash '+path)
checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
E=checkpoint['E'].numpy();R=checkpoint['R'].numpy()
check(E.shape==(11258,200) and R.shape==(474,200),'Parameter shapes')
check(np.isfinite(E).all() and np.isfinite(R).all(),'Finite checkpoint')
if job['method'] in ('RandomExpansion','NewEntityOnly','FastKGE','STARK'):
    check(np.array_equal(E[:7505],initial['oldE'].numpy()),'Method frozen old entities')
    check(np.array_equal(R,initial['R'].numpy()),'Method frozen relations')
fix=np.load(out/'audit-fixture.npz');q=fix['queries'];saved=fix['scores']
if job['kind']=='TransE':
    En=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-12)
    Rn=R/np.maximum(np.linalg.norm(R,axis=1,keepdims=True),1e-12)
    actual=np.stack([1/(1+np.exp(-(9-np.abs(En[h]+Rn[r]-En).sum(1)))) for h,r,t in q])
    tolerance=1e-12
else:
    actual=np.stack([(E[h]*R[r])@E.T for h,r,t in q]);tolerance=2e-4
error=float(np.abs(actual-saved).max());check(error<=tolerance,'Independent full candidate scores')
actual_rank=1+(actual>actual[np.arange(len(q)),q[:,2]][:,None]).sum(1)
saved_rank=1+(saved>saved[np.arange(len(q)),q[:,2]][:,None]).sum(1)
check(np.array_equal(actual_rank,saved_rank),'Fixture exact ranks')
table=np.load(out/'ranks.npz');summary=json.loads((out/'evaluation.json').read_text())
check(np.array_equal(q,np.concatenate([table['old_queries'][:32],table['new_queries'][:32]])),'Fixture query IDs')
check(np.array_equal(actual_rank,np.concatenate([table['old_after'][:32],table['new_after'][:32]])),'Fixture ranks vs complete export')
source=(root/cfg['source_root']/cfg['dataset']).resolve()
for s,name in [(0,'old'),(1,'new')]:
    mapping=dict((k,int(v)) for k,v in (line.split() for line in (source/str(s)/'entity2id.txt').read_text().splitlines()))
    relation=dict((k,int(v)) for k,v in (line.split() for line in (source/str(s)/'relation2id.txt').read_text().splitlines()))
    expected=[]
    for line in (source/str(s)/'test.txt').read_text().splitlines():
        h,r,t=line.split();h,r,t=mapping[h],relation[r],mapping[t];expected.extend([(h,r,t),(t,r+1,h)])
    check(np.array_equal(table[name+'_queries'],expected),'Full '+name+' TEST IDs')
for name in ['old_before','old_after','new_after']:
    rank=table[name]; check(((rank>=1)&(rank<=(7505 if name=='old_before' else 11258))).all(),name+' rank range')
    for metric,value in [('raw_mrr',(1/rank).mean()),('hits1',(rank<=1).mean()),('hits10',(rank<=10).mean())]:
        check(abs(summary[name][metric]-value)<1e-14,name+' '+metric)
mask=table['old_before']<=10;check(abs(summary['old_top10_loss']-(table['old_after'][mask]>10).mean())<1e-14,'Main Top10 loss')
mask=table['old_expanded']<=10;check(abs(summary['same_pool_top10_loss']-(table['old_after'][mask]>10).mean())<1e-14,'Expanded pool loss')
audit=dict(status='PASS_INFERENCE_AND_AGGREGATION' if not failures else 'FAIL_RETAINED',checks=checks,failures=failures,
           candidate_score_checks=int(actual.size),max_abs_score_error=error,fixture_rank_mismatches=int((actual_rank!=saved_rank).sum()),
           auditor_sha256=sha(Path(__file__)),limits=['64 complete-candidate rows; no independent full retraining or full-table score replay.'])
(out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));sys.exit(bool(failures))
