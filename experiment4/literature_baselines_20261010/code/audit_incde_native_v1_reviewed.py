import argparse,json,pathlib,hashlib,re,collections,copy,os
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  while chunk:=stream.read(4*1024*1024):h.update(chunk)
 return h.hexdigest()
def audit(run,device):
 failures=[];checks=0
 def ck(ok,message,n=1):
  nonlocal checks;checks+=n
  if not ok:failures.append(message)
 job=json.loads((run/'registered-job.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
 ck(terminal['exit_code']==0,'terminal exit');ck(terminal.get('source_inputs_unchanged') is True,'producer input check')
 for name,digest in job['input_sha256'].items():ck(sha(R/name)==digest,'original source/input:'+name)
 for name,digest in json.loads((run/'copied-source-sha256.json').read_text()).items():
  # Author generate_layers overwrites its own derived train_sorted file; original raw inputs remain immutable.
  if pathlib.Path(name).name=="train_sorted_by_edges_betweenness.txt":continue
  ck(sha(run/name)==digest,'copied source/input:'+name)
 entities={};relations={};snapshots=[];filters=collections.defaultdict(set)
 for ss in range(5):
  facts={}
  for split in ['train','valid','test']:
   rows=[tuple(x.split()) for x in (run/'worktree/data'/job['dataset']/str(ss)/(split+'.txt')).read_text().splitlines() if x.strip()]
   for h,r,t in rows:
    for e in [h,t]:
     if e not in entities:entities[e]=len(entities)
    if r not in relations:
     k=len(relations);relations[r]=k;relations[r+'_inv']=k+1
   facts[split]=[(entities[h],relations[r],entities[t]) for h,r,t in rows]
  for rows in facts.values():
   for h,r,t in rows:filters[(h,r)].add(t);filters[(t,r+1)].add(h)
  queries=[(h,r,t) for h,r,t in facts['test']]+[(t,r+1,h) for h,r,t in facts['test']]
  ck(set(facts['train']).isdisjoint(facts['valid']),'train/valid overlap snapshot'+str(ss))
  ck(set(facts['train']).isdisjoint(facts['test']),'train/test overlap snapshot'+str(ss))
  ck(set(facts['valid']).isdisjoint(facts['test']),'valid/test overlap snapshot'+str(ss))
  snapshots.append({'entities':len(entities),'relations':len(relations),'queries':queries,'filters':copy.deepcopy(dict(filters)),'test_triples':len(facts['test'])})
 files=sorted((run/'ranks').glob('*.npz'));ck(len(files)==19,'19 rank files');metrics=[];replayed=0
 expected={f'train{s}_test{t}_post_train.npz' for s in range(5) for t in range(s+1)}|{f'train{s}_test{s}_pre_fwt.npz' for s in range(1,5)}
 ck({p.name for p in files}==expected,'rank file identities')
 log=(run/'run.log').read_text();ck('Final Result:' in log and 'Report Result:' in log,'native final log')
 for s in range(5):
  epochs=[int(x) for x in re.findall(r'Snapshot:'+str(s)+r'\s+Epoch:(\d+)',log)]
  ck(bool(epochs) and max(epochs)>30,'full non-smoke training:'+str(s))
 ck(len(list((run/'checkpoint'/job['dataset']).glob('*model_best.tar')))==5,'five best checkpoints')
 for file in files:
  with np.load(file,allow_pickle=False) as z:data={k:z[k] for k in z.files}
  s=int(data['trained_snapshot']);t=int(data['tested_snapshot']);snap=snapshots[t];rank=data['rank'];n=len(rank)
  ck(rank.dtype.kind in 'iu','integer rank:'+file.name)
  ck(np.all((rank>=1)&(rank<=snap['entities'])),'rank range:'+file.name,n)
  ck(int(data['seed'])==3407,'seed:'+file.name)
  ck(np.all(data['candidates']==snap['entities']),'candidate set:'+file.name,n)
  q=list(zip(data['head'].tolist(),data['relation'].tolist(),data['tail'].tolist()))
  ck(collections.Counter(q)==collections.Counter(snap['queries']),'query identity:'+file.name,n)
  ck(np.array_equal(data['filtered_count'],[len(snap['filters'][(h,r)]) for h,r,tgt in q]),'filter count:'+file.name,n)
  raw={'mrr':float(np.mean(1/rank)),'hits1':float(np.mean(rank<=1)),'hits3':float(np.mean(rank<=3)),'hits10':float(np.mean(rank<=10))}
  metrics.append({'file':file.name,'trained_snapshot':s,'tested_snapshot':t,'phase':'pre_fwt' if 'pre_fwt' in file.name else 'post_train','queries':n,'raw':raw,'native_rounded':{k:round(v,4) for k,v in raw.items()}})
  if 'post_train' in file.name:
   checkpoint=run/'checkpoint'/job['dataset']/(str(s)+'model_best.tar');state=torch.load(checkpoint,map_location=device,weights_only=False)['state_dict']
   ent=torch.nn.functional.normalize(state['ent_embeddings.weight'],p=2,dim=-1);rel=torch.nn.functional.normalize(state['rel_embeddings.weight'],p=2,dim=-1)
   indices=np.unique(np.linspace(0,n-1,min(64,n)).astype(int));out=[]
   with torch.no_grad():
    for i in range(0,len(indices),16):
     ids=indices[i:i+16];qs=np.array([q[j] for j in ids]);heads=torch.as_tensor(qs[:,0],device=device);rels=torch.as_tensor(qs[:,1],device=device);tails=torch.as_tensor(qs[:,2],device=device)
     # Independent formula; preserves native float64, sigmoid, candidate set and tie convention.
     pred=torch.sigmoid(9-torch.norm((ent[heads]+rel[rels])[:,None,:]-ent[:snap['entities']][None,:,:],p=1,dim=2));row=torch.arange(len(ids),device=device);target=pred[row,tails].clone()
     for j,(h,r,tgt) in enumerate(qs.tolist()):pred[j,list(snap['filters'][(h,r)])]=-10000000
     pred[row,tails]=target;rerank=1+pred.argsort(dim=1,descending=True).argsort(dim=1)[row,tails];out.extend(rerank.cpu().tolist())
   ck(np.array_equal(out,rank[indices]),'independent checkpoint rank replay:'+file.name,len(indices));replayed+=len(indices)
   del state,ent,rel
 overall=[]
 for s in range(5):
  subset=[x for x in metrics if x['trained_snapshot']==s and x['phase']=='post_train'];count=sum(x['queries'] for x in subset)
  vals={k:round(sum(x['native_rounded'][k]*x['queries'] for x in subset)/count,3) for k in ['mrr','hits1','hits3','hits10']}
  overall.append({'snapshot':s,'queries':count,**vals})
  pattern=r'\|\s*'+str(s)+r'\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|'
  matches=re.findall(pattern,log);ck(bool(matches),'logged aggregate presence:'+str(s))
  if matches:ck(np.allclose(list(vals.values()),list(map(float,matches[-1])),rtol=0,atol=1e-12),'logged aggregate matches:'+str(s))
 digest={str(p.relative_to(run)):sha(p) for p in files+list((run/'checkpoint'/job['dataset']).glob('*model_best.tar'))+[run/'run.log',run/'registered-job.json',run/'terminal.json']}
 report={'id':job['id'],'method':'IncDE','dataset':job['dataset'],'status':'PASS' if not failures else 'FAIL','checks':checks,'failures':failures,'independent_post_train_rank_replays':replayed,'per_task_metrics':metrics,'native_overall':overall,'artifact_sha256':digest,'limitations':['single seed author protocol','19 metric exports audited; checkpoint ranks sampled64 per post-training file, not every score','pre-FWT exact checkpoint replay unavailable; counts/ranges/filters checked','historical query-snapshot candidates and filters do not account for later competitors','native float32 metric accumulation and task rounding retained']}
 out=R/'results/KGE-LIT-001-audit'/job['id'];out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['id','status','checks','failures','independent_post_train_rank_replays','native_overall']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=pathlib.Path);p.add_argument('--device',default='cuda:0');a=p.parse_args();audit(a.run.resolve(),a.device)
