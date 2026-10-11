"""TRAIN-only feasible-coordinate preflight and independent CPU fixtures."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common_training import sha
from common_incremental import Data
from shared_rank_scope import source_initial
from shared_rank_graph import load_scope
from shared_rank_cache_v2 import SharedRankCache,fixed_scores
from feasible_train_optimization import TrainIndex,optimize_train_row


def worker(root,cfg,job):
    torch.set_num_threads(2);started=time.monotonic();device=torch.device('cuda:0')
    initial,R,bound=source_initial(root,cfg,job['seed'],device)
    original=root/'artifacts/incremental'/job['initial_endpoint']
    audit=json.loads((original/'audit.json').read_text());assert not audit['failures'] and not audit['solver_audit']['failures']
    assert audit['solver_audit']['patches_replayed']==3753 and audit['solver_audit']['terminal_cpu_counts_exact']
    tr=json.loads((original/'training-receipt.json').read_text())
    assert sha(original/'best.pt')==tr['checkpoint_sha256']
    for name,digest in tr['bound_input_sha256'].items():assert sha(name)==digest,name
    model=torch.load(original/'best.pt',map_location=device,weights_only=True)
    E=model['E'].double();assert torch.equal(E[:7505],initial[:7505]) and torch.equal(model['R'].double(),R)
    for name in ('best.pt','audit.json','training-receipt.json'):bound[str(original/name)]=sha(original/name)
    keys,refs,scope_bound=load_scope(root,cfg,job['seed'],job['protection_scope']);bound.update(scope_bound)
    cache=SharedRankCache(E,R,keys,refs,cfg['k'],cfg['score_batch'])
    data=Data(root,cfg);old=data.triples(0,'train').numpy();new=data.triples(1,'train').numpy()
    index=TrainIndex(old,new);radius=2*float(torch.linalg.vector_norm(initial,dim=1).max().cpu())
    order=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy()[:cfg['updates']]
    out=root/cfg['output_directory']/job['id'];out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(E=E.cpu(),R=R.cpu(),centers=initial.cpu()),out/'initial.pt')
    before=[];after=[];positive=[];negative=[];records=[]
    for i,e in enumerate(order):
        assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Frozen preflight budget exceeded'
        e=int(e);before.append(cache.E[e].cpu().numpy().copy())
        patch,receipt,p,n=optimize_train_row(cache,index,e,initial[e].cpu().numpy(),radius,cfg['distmult_l2'],cfg['negatives'],cfg['solver'])
        if receipt['accepted']:assert cache.commit(patch)
        else:assert np.array_equal(patch.rows.cpu().numpy()[0],before[-1])
        after.append(cache.E[e].cpu().numpy().copy());positive.append(p);negative.append(n)
        records.append(dict(entity=e,receipt=receipt,full_scope_max_rank=int(cache.counts.max())+1,
                            state_sha256=hashlib.sha256(cache.E.cpu().numpy().tobytes()).hexdigest()))
        print('FEASIBLE_TRAIN_PROGRESS',job['id'],i+1,len(order),receipt['accepted'],round(receipt['before_loss']-receipt['after_loss'],6),flush=True)
    terminal=cache.full_recount();assert torch.equal(cache.E[:7505],initial[:7505]) and torch.equal(cache.R,R)
    fixture_ids=np.sort(np.random.default_rng(cfg['fixture_seed']+job['seed']).choice(len(keys),cfg['fixture_queries'],replace=False))
    fixtures=cache.S[torch.as_tensor(fixture_ids,device=device)].cpu().numpy()
    np.savez_compressed(out/'replay.npz',order=order,before=before,after=after,fixture_ids=fixture_ids,
                        fixture_scores=fixtures,terminal_counts=cache.counts.cpu().numpy(),
                        **{f'{name}_{i}':rows for name,parts in [('positive',positive),('negative',negative)] for i,rows in enumerate(parts)})
    torch.save(dict(E=cache.E.cpu(),R=R.cpu()),out/'best.pt')
    (out/'steps.json').write_text(json.dumps(records,indent=2))
    bound.update(data.bound)
    for name,digest in bound.items():assert sha(name)==digest,name
    receipt=dict(id=job['id'],seed=job['seed'],scope='TRAIN-only direct feasible loss preflight; no graph utility claim',
         updates=len(order),accepted=sum(r['receipt']['accepted'] for r in records),initial_endpoint=job['initial_endpoint'],
         endpoint_elapsed_s=tr['elapsed_s'],source_sgd_elapsed_s=tr['source_sgd_elapsed_s'],
         elapsed_s=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),gpu_name=torch.cuda.get_device_name(),
         radius=radius,terminal=terminal,valid_queries_read=False,test_queries_read=False,
         input_sha256=bound,output_sha256={name:sha(out/name) for name in ['initial.pt','best.pt','steps.json','replay.npz']},
         implementation_sha256=sha(Path(__file__)),module_sha256=sha(root/'src/feasible_train_optimization.py'),
         limits=['Local fixed-hard-negative loss is not full TRAIN/global/VALID utility.',
                 'All original full/scoped references unchanged; immutable-old barriers remain.',
                 'Independent CPU fixture coverage is bounded; complete scope GPU recount only in preflight.'])
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('FEASIBLE_TRAIN_COMPLETE',job['id'],receipt['accepted'],flush=True)


def numpy_scores(E,R,key):
    h,r=key;tree=np.zeros((len(E),256));tree[:,:200]=E*(E[h]*R[r])
    while tree.shape[1]>1:tree=tree[:,::2]+tree[:,1::2]
    return tree[:,0]


def scalar_loss(E,R,positive,negative,l2,entity):
    p=np.array([np.sum(E[h]*R[r]*E[t]) for h,r,t in positive])
    n=np.array([np.sum(E[h]*R[r]*E[t]) for h,r,t in negative])
    return float(np.logaddexp(0,-p).mean()+np.logaddexp(0,n).mean()+l2*np.mean(E[entity]**2))


def audit(root,cfg,job):
    started=time.monotonic();torch.set_num_threads(2);out=root/cfg['output_directory']/job['id']
    receipt=json.loads((out/'receipt.json').read_text());assert not receipt['valid_queries_read'] and not receipt['test_queries_read']
    for name,digest in receipt['input_sha256'].items():assert sha(name)==digest,name
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest,name
    model=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
    E=model['E'].numpy().copy();R=model['R'].numpy();centers=model['centers'].numpy()
    data=Data(root,cfg);old=data.triples(0,'train').numpy();new=data.triples(1,'train').numpy()
    truth={}
    for h,r,t in np.r_[old,new]:truth.setdefault((int(h),int(r)),set()).add(int(t))
    keys,refs,_=load_scope(root,cfg,job['seed'],job['protection_scope'])
    rows=json.loads((out/'steps.json').read_text());scalar_checks=0;max_loss_error=0.
    with np.load(out/'replay.npz') as replay:
        expected=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy()[:cfg['updates']]
        assert np.array_equal(replay['order'],expected) and len(rows)==len(expected)
        for i,e in enumerate(expected):
            e=int(e);assert rows[i]['entity']==e and np.array_equal(E[e],replay['before'][i])
            p=replay[f'positive_{i}'];n=replay[f'negative_{i}']
            selected=new[(new[:,0]==e)|(new[:,2]==e)]
            assert np.array_equal(p,selected)
            want=[]
            for h,r,t in p:
                scores=numpy_scores(E,R,(h,r));scores[list(truth[(int(h),int(r))])]=-np.inf
                wanted=np.argsort(-scores,kind='stable')[:cfg['negatives']]
                want.extend((int(h),int(r),int(c)) for c in wanted)
            assert np.array_equal(n,want),'Independent full-pool hardest TRAIN-filtered competitors differ'
            before=scalar_loss(E,R,p,n,cfg['distmult_l2'],e)
            E[e]=replay['after'][i]
            after=scalar_loss(E,R,p,n,cfg['distmult_l2'],e)
            for value,name in ((before,'before_loss'),(after,'after_loss')):
                error=abs(value-rows[i]['receipt'][name]);max_loss_error=max(max_loss_error,error)
                assert error<1e-10;scalar_checks+=1
            assert after<=before+1e-10 and np.linalg.norm(E[e]-centers[e])<=receipt['radius']+cfg['solver']['movement_tolerance']
            if not rows[i]['receipt']['accepted']:assert np.array_equal(replay['before'][i],replay['after'][i])
            assert hashlib.sha256(E.tobytes()).hexdigest()==rows[i]['state_sha256']
        final=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        assert np.array_equal(E,final['E'].numpy()) and np.array_equal(R,final['R'].numpy())
        assert np.array_equal(E[:7505],centers[:7505])
        cells=0
        for fi,qi in enumerate(replay['fixture_ids']):
            scores=numpy_scores(E,R,keys[qi]);assert np.array_equal(scores.view(np.uint64),replay['fixture_scores'][fi].view(np.uint64))
            members=refs[refs[:,0]==qi,1];count=int((scores>min(scores[members])).sum())
            assert count==replay['terminal_counts'][qi] and count<cfg['k'];cells+=len(scores)
    result=dict(status='PASS_ALL_PREFLIGHT_PATCHES_TRAIN_MINING_LOSS_AND_FULL_POOL_CPU_FIXTURES',failures=[],
         scalar_loss_checks=scalar_checks,max_loss_error=max_loss_error,score_cells_replayed=cells,
         patches_replayed=len(rows),old_rows_and_relations_unchanged=True,guard_member_identities_unchanged=True,
         receipt_sha256=sha(out/'receipt.json'),replay_sha256=sha(out/'replay.npz'),elapsed_s=time.monotonic()-started,
         valid_queries_read=False,test_queries_read=False,scientific_utility_asserted=False,
         limits=['Independent32 full query CPU rows per job; complete terminal scope GPU only at preflight.'])
    assert not (out/'audit.json').exists();(out/'audit.json').write_text(json.dumps(result,indent=2))
    print('FEASIBLE_TRAIN_AUDIT_PASS',job['id'],cells,flush=True)


def coordinate(root,cfg,config):
    canonical=root/'experiments'/cfg['id']/'experiment.yaml';state=yaml.safe_load(canonical.read_text());assert state['status']=='registered'
    for name,digest in state['frozen_input_sha256'].items():assert sha(root/name)==digest,name
    idle_output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle={int(i) for line in idle_output.splitlines() for i,m,u in [line.split(',')] if int(m)<=1024 and int(u)<=20}
    assert set(cfg['gpu_indices'])<=idle
    state.update(status='running_train_only_preflight',coordinator_pid=os.getpid(),jobs={});active={}
    def save():
        state['active_processes']=[dict(id=j,pid=x['proc'].pid,stage=x['stage'],gpu=x['gpu'],start_ticks=x['ticks']) for j,x in active.items()]
        tmp=canonical.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(canonical)
    def start(job,stage,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if stage=='audit' else str(gpu),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(root/cfg['log_directory']/f"{job['id']}-{stage}.log").open('xb')
        command=[os.path.abspath(root/cfg['python']),'-u',str(Path(__file__)),'--root',str(root),'--config',config,'--stage',stage,'--job',job['id']]
        proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]
        active[job['id']]=dict(proc=proc,log=log,stage=stage,gpu=gpu,job=job,ticks=ticks)
        state['jobs'][job['id']]=dict(status='running',stage=stage,pid=proc.pid)
        print('FEASIBLE_TRAIN_START',job['id'],stage,proc.pid,gpu,flush=True)
    for job,gpu in zip(cfg['jobs'],cfg['gpu_indices']):start(job,'run',gpu)
    save()
    while active:
        for name,row in list(active.items()):
            rc=row['proc'].poll()
            if rc is None:continue
            row['log'].close();del active[name]
            if rc==0 and row['stage']=='run':start(row['job'],'audit',None)
            else:state['jobs'][name]=dict(status='completed_audited' if rc==0 else 'failed_retained',stage=row['stage'],exit_code=rc)
        save()
        if active:time.sleep(3)
    dest=root/'results'/cfg['id'];dest.mkdir(exist_ok=False)
    results=[]
    for job in cfg['jobs']:
        if state['jobs'][job['id']]['status']=='completed_audited':
            out=root/cfg['output_directory']/job['id'];results.append(dict(job=job,receipt=json.loads((out/'receipt.json').read_text()),audit=json.loads((out/'audit.json').read_text())))
    (dest/'summary.json').write_text(json.dumps(dict(status='PASS_ALL_SIX_TRAIN_ONLY_PREFLIGHTS' if len(results)==6 else 'RETAINED_PREFLIGHT_FAILURES',rows=results,
              completed_audited_jobs=len(results),planned_jobs=6,scientific_utility_asserted=False),indent=2))
    state.update(status='completed_audited_train_preflight_only' if len(results)==6 else 'terminal_with_retained_failures',result_directory=str(dest.relative_to(root)),completed_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',default='configs/feasible-train40.json');p.add_argument('--stage',choices=['run','audit','coordinate'],required=True);p.add_argument('--job')
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text())
    if a.stage=='coordinate':coordinate(root,cfg,a.config)
    elif a.stage=='run':worker(root,cfg,next(j for j in cfg['jobs'] if j['id']==a.job))
    else:audit(root,cfg,next(j for j in cfg['jobs'] if j['id']==a.job))
