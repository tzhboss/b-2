import pathlib,argparse,json,hashlib,time,collections,copy
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parents[1]
NAMES=['native_replay','candidate_expansion_only','strict_rank_old_candidates','strict_rank_final_candidates','current_filter_final_candidates']
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*1024*1024):h.update(x)
 return h.hexdigest()
def main(run):
 job=json.loads((run/'registered-job.json').read_text());audit_path=R/'results/KGE-LIT-001-audit'/job['id']/'audit.json';audit=json.loads(audit_path.read_text());assert audit['status']=='PASS'
 protocol=R/'configs/protocols/KGE-LIT-005-candidate-diagnostic.json';assert file_sha(protocol)=='e36e51c12d983f5875d326e94f3755d2dccf8b3d39af719a76daacabdd79111a'
 out=R/'runs/candidate-diagnostic'/job['id'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();device='cuda:0'
 entids={};relids={};filters=collections.defaultdict(set);snapshots=[]
 for ss in range(5):
  facts={}
  for split in ['train','valid','test']:
   rows=[line.split() for line in (run/'worktree/data'/job['dataset']/str(ss)/(split+'.txt')).read_text().splitlines() if line.strip()]
   for h,r,t in rows:
    for e in [h,t]:
     if e not in entids:entids[e]=len(entids)
    if r not in relids:k=len(relids);relids[r]=k;relids[r+'_inv']=k+1
   facts[split]=[(entids[h],relids[r],entids[t]) for h,r,t in rows]
  for rows in facts.values():
   for h,r,t in rows:filters[(h,r)].add(t);filters[(t,r+1)].add(h)
  snapshots.append({'entities':len(entids),'filters':copy.deepcopy(dict(filters))})
 ckpt=run/'checkpoint'/job['dataset']/'4model_best.tar';checkpoint=torch.load(ckpt,map_location=device,weights_only=False);state=checkpoint['state_dict'];ent=torch.nn.functional.normalize(state['ent_embeddings.weight'],p=2,dim=-1);rel=torch.nn.functional.normalize(state['rel_embeddings.weight'],p=2,dim=-1);assert len(ent)==snapshots[4]['entities']
 checks=0;table=[];output_hashes={}
 with torch.no_grad():
  for t in range(5):
   source=run/'ranks'/f'train4_test{t}_post_train.npz'
   with np.load(source,allow_pickle=False) as z:q=np.column_stack([z['head'],z['relation'],z['tail']]);native_rank=z['rank'].copy()
   limit=snapshots[t]['entities'];olds=snapshots[t]['filters'];new=snapshots[4]['filters'];allr={name:[] for name in NAMES};ties=[];later=[]
   for i in range(0,len(q),32):
    b=q[i:i+32];ids=torch.as_tensor(b,device=device);idx=torch.arange(len(b),device=device);target_tail=ids[:,2]
    score=torch.sigmoid(9-torch.norm((ent[ids[:,0]]+rel[ids[:,1]])[:,None,:]-ent[None,:,:],p=1,dim=2));target=score[idx,target_tail].clone()
    for j,(h,r,tail) in enumerate(b.tolist()):score[j,list(olds[(h,r)])]=-10000000
    score[idx,target_tail]=target
    old=score[:,:limit];oldrank=1+old.argsort(dim=1,descending=True).argsort(dim=1)[idx,target_tail];finalrank=1+score.argsort(dim=1,descending=True).argsort(dim=1)[idx,target_tail]
    strictold=1+(old>target[:,None]).sum(1);strictfinal=1+(score>target[:,None]).sum(1)
    assert torch.all(strictfinal>=strictold);checks+=len(b)
    ties.extend(((score==target[:,None]).sum(1)-1).cpu().tolist())
    for j,(h,r,tail) in enumerate(b.tolist()):
     later.append(len(new[(h,r)]-olds[(h,r)]));score[j,list(new[(h,r)])]=-10000000
    score[idx,target_tail]=target;currentrank=1+(score>target[:,None]).sum(1)
    for name,val in zip(NAMES,[oldrank,finalrank,strictold,strictfinal,currentrank]):allr[name].extend(val.cpu().tolist())
    if i%6400==0:print('PROGRESS',job['dataset'],t,i,len(q),flush=True)
   ranks={k:np.asarray(v,dtype=np.int64) for k,v in allr.items()};assert np.array_equal(ranks['native_replay'],native_rank),'all-query native replay mismatch';checks+=len(q)
   npz=out/f'test{t}.npz';np.savez_compressed(npz,head=q[:,0],relation=q[:,1],tail=q[:,2],**ranks,equal_score_other_candidates=np.asarray(ties,dtype=np.int64),later_filtered_truths=np.asarray(later,dtype=np.int64));output_hashes[npz.name]=file_sha(npz)
   summaries={name:{'mrr':float(np.mean(1/r)),'hits1':float(np.mean(r<=1)),'hits3':float(np.mean(r<=3)),'hits10':float(np.mean(r<=10))} for name,r in ranks.items()}
   table.append({'task_snapshot':t,'queries':len(q),'old_candidates':limit,'final_candidates':len(ent),'scores':summaries,'strict_candidate_rank_worse_fraction':float(np.mean(ranks['strict_rank_final_candidates']>ranks['strict_rank_old_candidates'])),'native_candidate_rank_worse_fraction':float(np.mean(ranks['candidate_expansion_only']>ranks['native_replay'])),'tie_fraction':float(np.mean(np.asarray(ties)>0)),'queries_with_later_filtered_truths':int(np.sum(np.asarray(later)>0))})
 total=sum(x['queries'] for x in table);overall={name:{metric:sum(x['scores'][name][metric]*x['queries'] for x in table)/total for metric in ['mrr','hits1','hits3','hits10']} for name in NAMES}
 result={'id':job['id'],'status':'terminal_unreviewed','registration_sha256':file_sha(protocol),'source_audit_sha256':file_sha(audit_path),'checkpoint_sha256':file_sha(ckpt),'full_process_elapsed_s':time.monotonic()-start,'queries':total,'checks':checks,'all_query_native_replay_exact':True,'task_table':table,'overall':overall,'output_sha256':output_hashes,'implementation_sha256':file_sha(pathlib.Path(__file__))};(out/'terminal.json').write_text(json.dumps(result,indent=2));print('DIAGNOSTIC_TERMINAL',job['id'],total,overall,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=pathlib.Path);a=p.parse_args();main(a.run.resolve())
