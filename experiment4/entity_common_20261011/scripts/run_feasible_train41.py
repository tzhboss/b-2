"""Full constrained TRAIN optimization, common inference and independent audits."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import run_feasible_train40 as original
from run_feasible_train_v2 import worker as v2worker
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common_training import sha
from common_incremental import Data,ranks,metrics,scores
from shared_rank_graph import load_scope
from audit_shared_graph_utility import numpy_ranks


def validation(root,cfg,job,E,R,initial,source,numpy=False):
    data=Data(root,cfg);vg=torch.Generator().manual_seed(cfg['validation_seed']);queries=[]
    for snapshot in (0,1):
        q=data.triples(snapshot,'valid');queries.append(q[torch.randperm(len(q),generator=vg)[:cfg['validation_queries_per_split']]])
    if numpy:
        queries=[q.numpy() for q in queries]
        compute=lambda model,q:numpy_ranks(model,R,q,64)
    else:compute=lambda model,q:ranks(model,R,q,'DistMult')
    sr=[compute(source,q) for q in queries];ar=[compute(E,q) for q in queries]
    old=compute(initial[:7505],queries[0]);good=old<=10
    q=queries[1];masks={'query_head_new_gold_old':(q[:,0]>=7505)&(q[:,2]<7505),
         'query_head_old_gold_new':(q[:,0]<7505)&(q[:,2]>=7505),'both_new':(q[:,0]>=7505)&(q[:,2]>=7505)}
    if not numpy:masks={k:v.numpy() for k,v in masks.items()}
    result=dict(source=[metrics(r) for r in sr],after=[metrics(r) for r in ar],
       source_old_top10_loss=float((sr[0][good]>10).mean()),after_old_top10_loss=float((ar[0][good]>10).mean()),
       directional={k:dict(queries=int(m.sum()),source=metrics(sr[1][m]),after=metrics(ar[1][m])) for k,m in masks.items()})
    return result,queries,data.bound


def train(root,cfg,job):
    assert cfg['updates']==3753
    v2worker(root,cfg,job)
    out=root/cfg['output_directory']/job['id'];receipt=json.loads((out/'receipt.json').read_text())
    model=torch.load(out/'best.pt',map_location='cuda:0',weights_only=True)
    initial=torch.load(out/'initial.pt',map_location='cuda:0',weights_only=True)
    source=torch.load(root/'artifacts/incremental'/cfg['source_template'].format(seed=job['seed'])/'best.pt',map_location='cuda:0',weights_only=True)
    val,queries,bound=validation(root,cfg,job,model['E'],model['R'],initial['centers'],source['E'].double())
    receipt['scope']='Full3753-entity direct TRAIN optimization inside original unchanged rank guards; fixed endpoint'
    receipt['validation']=val;receipt['input_sha256'].update(bound)
    receipt['valid_queries_read']=True;receipt['test_queries_read']=False
    receipt['single_fixed_endpoint_no_valid_selection']=True
    receipt['full_run_wrapper_sha256']=sha(Path(__file__))
    receipt['input_sha256'].update({str(root/name):sha(root/name) for name in ['scripts/run_feasible_train41.py',cfg['config_path']]})
    parent=json.loads((root/'artifacts/incremental'/job['initial_endpoint']/'training-receipt.json').read_text())
    preparation=json.loads((root/cfg['scope_directory']/f"shared-rank-scope-s{job['seed']}"/'receipt.json').read_text())['elapsed_s']
    receipt['source_inclusive_incremental_elapsed_s']=parent['source_sgd_elapsed_s']+preparation+parent['elapsed_s']+receipt['elapsed_s']
    receipt['source_inclusive_cost_components']=dict(source_sgd=parent['source_sgd_elapsed_s'],original_scope_preparation=preparation,original_qp=parent['elapsed_s'],new_optimization=receipt['elapsed_s'],
         separate_validation_and_audits_not_in_training_cost=True)
    receipt['limits']=[s for s in receipt['limits'] if 'preflight' not in s.lower()]
    receipt['limits'].append('Complete fixed-VALID is evaluated after one fixed endpoint; no checkpoint or utility-based search.')
    receipt['output_sha256']={name:sha(out/name) for name in ['initial.pt','best.pt','steps.json','replay.npz']}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    tr=dict(receipt,test_read=False,preflight=False,method=job['method'],kind='DistMult',
            checkpoint_sha256=sha(out/'best.pt'),bound_input_sha256=receipt['input_sha256'])
    (out/'training-receipt.json').write_text(json.dumps(tr,indent=2))
    print('FULL_TRAIN41_FIXED_VALID',job['id'],json.dumps(val),flush=True)


@torch.no_grad()
def evaluate(root,cfg,job):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic()
    out=root/cfg['output_directory']/job['id'];model=torch.load(out/'best.pt',map_location='cuda:0',weights_only=True)
    init=torch.load(out/'initial.pt',map_location='cuda:0',weights_only=True);initial=init['centers'];E,R=model['E'],model['R']
    data=Data(root,cfg);q=[data.triples(s,'test') for s in (0,1)]
    before=ranks(initial[:7505],R,q[0],'DistMult');expanded=ranks(initial,R,q[0],'DistMult')
    after=ranks(E,R,q[0],'DistMult');new=ranks(E,R,q[1],'DistMult');good=before<=10;expanded_good=expanded<=10
    np.savez_compressed(out/'ranks.npz',old_queries=q[0].numpy(),new_queries=q[1].numpy(),old_before=before,old_expanded=expanded,old_after=after,new_after=new)
    result=dict(id=job['id'],method=job['method'],seed=job['seed'],kind='DistMult',candidates=11258,
       old_before=metrics(before),old_after=metrics(after),new_after=metrics(new),old_top10_loss=float((after[good]>10).mean()),old_top10_protected=int(good.sum()),
       same_pool_top10_loss=float((after[expanded_good]>10).mean()),mean_rank_displacement=float((after-before).mean()),
       old_parameter_l2=float(torch.linalg.vector_norm(E[:7505]-initial[:7505]).cpu()),relation_parameter_l2=float(torch.linalg.vector_norm(R-init['R']).cpu()),
       evaluation_elapsed_s=time.monotonic()-started)
    fq=torch.cat([q[0][:32],q[1][:32]]);fixture=torch.cat([scores(E,R,x.to('cuda:0'),'DistMult').cpu() for x in fq.split(8)]).numpy()
    np.savez_compressed(out/'audit-fixture.npz',queries=fq.numpy(),scores=fixture)
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    (out/'evaluation-receipt.json').write_text(json.dumps(dict(test_input_sha256=data.bound,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))
    print('FULL_COMMON_TEST41_COMPLETE',job['id'],json.dumps(result),flush=True)


def audit(root,cfg,job):
    # Original040 auditor proves no VALID access; full fitting explicitly
    # evaluates a fixed VALID endpoint. Use a separate full-run audit below
    # rather than weakening or monkey-patching that preflight invariant.
    torch.set_num_threads(2);started=time.monotonic();out=root/cfg['output_directory']/job['id']
    receipt=json.loads((out/'receipt.json').read_text())
    assert receipt['updates']==3753 and not receipt['test_queries_read'] and receipt['single_fixed_endpoint_no_valid_selection']
    for name,digest in receipt['input_sha256'].items():assert sha(name)==digest,name
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest,name
    model=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
    E=model['E'].numpy().copy();R=model['R'].numpy();centers=model['centers'].numpy()
    data=Data(root,cfg);old=data.triples(0,'train').numpy();new=data.triples(1,'train').numpy();truth={};incidence={}
    for h,r,t in np.r_[old,new]:truth.setdefault((int(h),int(r)),set()).add(int(t))
    for i,(h,r,t) in enumerate(new):
        for entity in set([int(h),int(t)]):
            if entity>=7505:incidence.setdefault(entity,[]).append(i)
    rows=json.loads((out/'steps.json').read_text());loss_checks=0;max_error=0.;accepted=0
    with np.load(out/'replay.npz') as replay:
        expected=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy()
        assert np.array_equal(replay['order'],expected) and len(rows)==3753
        for i,e in enumerate(expected):
            e=int(e);assert rows[i]['entity']==e and np.array_equal(E[e],replay['before'][i])
            p=replay[f'positive_{i}'];n=replay[f'negative_{i}'];assert np.array_equal(p,new[incidence[e]])
            chosen={}
            for h,r in np.unique(p[:,:2],axis=0):
                s=original.numpy_scores(E,R,(h,r));s[list(truth[(int(h),int(r))])]=-np.inf
                chosen[(int(h),int(r))]=np.argsort(-s,kind='stable')[:cfg['negatives']]
            wanted=np.asarray([(int(h),int(r),int(c)) for h,r,t in p for c in chosen[(int(h),int(r))]])
            assert np.array_equal(n,wanted),'Independent TRAIN mining differs'
            before=original.scalar_loss(E,R,p,n,cfg['distmult_l2'],e);E[e]=replay['after'][i]
            after=original.scalar_loss(E,R,p,n,cfg['distmult_l2'],e)
            for v,label in ((before,'before_loss'),(after,'after_loss')):
                error=abs(v-rows[i]['receipt'][label]);max_error=max(max_error,error);assert error<1e-10;loss_checks+=1
            assert after<=before+1e-10 and np.linalg.norm(E[e]-centers[e])<=receipt['radius']+cfg['solver']['movement_tolerance']
            if not rows[i]['receipt']['accepted']:assert np.array_equal(replay['before'][i],replay['after'][i])
            accepted+=int(rows[i]['receipt']['accepted'])
            assert __import__('hashlib').sha256(E.tobytes()).hexdigest()==rows[i]['state_sha256'] and rows[i]['full_scope_max_rank']<=10
            if (i+1)%128==0:print('FULL_TRAIN41_CPU_PATCHES',job['id'],i+1,3753,round(time.monotonic()-started,1),flush=True)
        final=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        assert np.array_equal(E,final['E'].numpy()) and np.array_equal(R,final['R'].numpy()) and np.array_equal(E[:7505],centers[:7505])
        assert accepted==receipt['accepted']
        keys,refs,_=load_scope(root,cfg,job['seed'],job['protection_scope']);members={}
        for qi,g in refs:members.setdefault(int(qi),[]).append(int(g))
        counts=[];cells=0
        for qi,key in enumerate(keys):
            s=original.numpy_scores(E,R,key);count=int((s>min(s[members[qi]])).sum());assert count<cfg['k'];counts.append(count);cells+=len(s)
            if (qi+1)%8192==0:print('FULL_TRAIN41_CPU_SCOPE',job['id'],qi+1,len(keys),round(time.monotonic()-started,1),flush=True)
        assert np.array_equal(counts,replay['terminal_counts'])
        for i,qi in enumerate(replay['fixture_ids']):assert np.array_equal(original.numpy_scores(E,R,keys[qi]),replay['fixture_scores'][i])
    source=torch.load(root/'artifacts/incremental'/cfg['source_template'].format(seed=job['seed'])/'best.pt',map_location='cpu',weights_only=True)
    val,queries,vbound=validation(root,cfg,job,E,R,centers,source['E'].numpy().astype(np.float64),numpy=True)
    assert val==receipt['validation'],'Independent fixed VALID differs'
    train_audit=dict(status='PASS_ALL3753_TRAIN_PATCHES_FULL_CPU_TERMINAL_SCOPE_AND_FIXED_VALID',failures=[],
       patches_replayed=3753,loss_checks=loss_checks,max_loss_error=max_error,full_terminal_cpu_score_cells=cells,
       terminal_cpu_counts_exact=True,reference_members_retained=len(refs),query_groups=len(keys),old_rows_and_relations_unchanged=True,
       source_sha256=sha(out/'receipt.json'),elapsed_s=time.monotonic()-started)
    (out/'train-audit.json').write_text(json.dumps(train_audit,indent=2))
    (out/'utility.json').write_text(json.dumps(dict(validation=val,query_sha256=[__import__('hashlib').sha256(q.tobytes()).hexdigest() for q in queries],bound=vbound),indent=2))
    subprocess.run([sys.executable,str(root/'scripts/audit_exported_embeddings.py'),'--root',str(root),'--config',cfg['config_path'],'--job',job['id']],check=True)
    infer=json.loads((out/'audit.json').read_text());assert not infer['failures']
    infer['train_audit']=train_audit;infer['status']='PASS_FULL_COMMON_INFERENCE_TRAIN_OBJECTIVE_AND_FULL_CPU_GUARDS'
    (out/'audit.json').write_text(json.dumps(infer,indent=2));print('FULL_TRAIN41_AUDIT_COMPLETE',job['id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['run','evaluate','audit'],required=True);p.add_argument('--job',required=True)
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());assert cfg['config_path']==a.config
    job=next(j for j in cfg['jobs'] if j['id']==a.job)
    {'run':train,'evaluate':evaluate,'audit':audit}[a.stage](root,cfg,job)
