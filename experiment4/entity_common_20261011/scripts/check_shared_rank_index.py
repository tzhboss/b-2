"""Config-driven static-index equivalence controls and saved-model audit."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from hashlib import sha256
import numpy as np
import torch
import yaml


def digest(p):return sha256(p.read_bytes()).hexdigest()


def prepare(root,cfg,job):
    from shared_rank_cache_v2 import SharedRankCache,fixed_scores
    from shared_rank_solver_v2 import solve_shared_update as original
    from shared_rank_solver_v3 import solve_shared_update as cached,static_index,StaticGuardIndex
    rng=np.random.default_rng(cfg['seed']);out=root/cfg['reaudit_directory']/job['id'];out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();records=[];models=[];relations=[];all_refs=[];accepted=rejected=cuts=0
    n=cfg['entities'];d=cfg['dimension'];keys=np.array([(h,r) for h in range(n-1) for r in range(2)])
    for case in range(cfg['cases']):
        E=rng.normal(size=(n,d));R=rng.normal(size=(2,d))
        if case%8==0:E[1]=E[0]
        scores=fixed_scores(torch.from_numpy(E[keys[:,0]]*R[keys[:,1]]),torch.from_numpy(E)).numpy()
        refs=np.array([(q,int(g)) for q,row in enumerate(scores) for g in np.argsort(-row,kind='stable')[:2]])
        a=SharedRankCache(torch.from_numpy(E),torch.from_numpy(R),keys,refs,cfg['k'],8)
        b=SharedRankCache(torch.from_numpy(E),torch.from_numpy(R),keys,refs,cfg['k'],8)
        trajectory=[E.copy()];per_step=[];context=None
        for step in range(cfg['steps']):
            assert time.monotonic()-started<cfg['budget_seconds'],'Engineering control budget exceeded'
            ids=[n-1] if case%4==0 else [step%(n-1),n-1]
            target=b.E[ids].numpy().copy()+rng.normal(size=(len(ids),d))*(.03 if step%2==0 else .8)
            weights=[1.] if len(ids)==1 else [4.,1.];radii=np.full(len(ids),1.2);opts=dict(cfg['solver'])
            if case%7==0 and step==1:opts['max_cut_rounds']=0
            pa,ia=original(a,ids,target,weights,E[ids],radii,ids[:-1],opts)
            pb,ib=cached(b,ids,target,weights,E[ids],radii,ids[:-1],opts)
            assert {k:v for k,v in ia.items() if k!='elapsed_s'}=={k:v for k,v in ib.items() if k!='elapsed_s'},(case,step,ia,ib)
            for name in ['ids','rows','columns','head_queries','head_directions','head_scores','thresholds','counts']:
                assert torch.equal(getattr(pa,name),getattr(pb,name)),(case,step,name)
            if ia['accepted']:assert a.commit(pa) and b.commit(pb);accepted+=1
            else:assert np.array_equal(pa.rows.numpy(),a.E[ids].numpy());rejected+=1
            a.full_recount();b.full_recount();assert torch.equal(a.E,b.E) and torch.equal(a.S,b.S)
            index=static_index(b)
            if context is None:context=index
            else:assert index is context
            cuts+=ia['cuts'];trajectory.append(b.E.numpy().copy());per_step.append(dict(ids=ids,accepted=ia['accepted'],radii=radii.tolist()))
        # Static scope mutation must not silently reuse an outdated grouping.
        b.keys[0,0]=(int(b.keys[0,0])+1)%(n-1)
        try:static_index(b)
        except AssertionError:pass
        else:raise AssertionError('Changed static query scope accepted')
        b.keys.copy_(torch.from_numpy(keys));b.references[0,1]=(int(b.references[0,1])+1)%n
        try:static_index(b)
        except AssertionError:pass
        else:raise AssertionError('Changed static reference scope accepted')
        models.append(trajectory);relations.append(R);all_refs.append(refs);records.append(dict(case=case,steps=per_step))
    assert accepted and rejected and cuts,(accepted,rejected,cuts)
    scope=root/cfg['benchmark_scope'];input_hash=digest(scope)
    with np.load(scope) as z:bk=z['full_keys'].copy();br=z['full_refs'].copy()
    class Scope:pass
    bench=Scope();bench.keys=torch.from_numpy(bk);bench.references=torch.from_numpy(br)
    t=time.monotonic();index=StaticGuardIndex(bench);construction=time.monotonic()-t;uncached=[];validated=[]
    for _ in range(cfg['benchmark_repetitions']):
        t=time.monotonic();groups={}
        for q,g in br:groups.setdefault(int(q),[]).append(int(g))
        cpu_keys=bk.copy();uncached.append(time.monotonic()-t)
        assert {q:tuple(gs) for q,gs in groups.items()}==index.refs and np.array_equal(cpu_keys,index.keys)
        t=time.monotonic();index.validate(bench);validated.append(time.monotonic()-t)
    np.savez_compressed(out/'models.npz',models=np.asarray(models),relations=np.asarray(relations),keys=keys,refs=np.asarray(all_refs))
    (out/'steps.json').write_text(json.dumps(records))
    result=dict(status='PASS_V2_V3_BITWISE_PATCH_TRAJECTORIES',cases=cfg['cases'],steps=cfg['cases']*cfg['steps'],accepted=accepted,rejected=rejected,cuts=cuts,
                index_reused_within_trajectory=True,query_scope_mutation_rejected=True,reference_scope_mutation_rejected=True,test_read=False,valid_read=False,graph_utility_measured=False,
                benchmark=dict(query_keys=len(bk),reference_members=len(br),one_time_index_s=construction,uncached_grouping_s=uncached,cached_validation_s=validated,
                               scope_sha256=input_hash,limits=['Prepared TRAIN index construction only, CPU timing; no full solver or GPU speedup inference.']),
                models_sha256=digest(out/'models.npz'),steps_sha256=digest(out/'steps.json'),elapsed_s=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def audit(root,cfg,job):
    out=root/cfg['reaudit_directory']/job['id'];receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='PASS_V2_V3_BITWISE_PATCH_TRAJECTORIES'
    assert digest(out/'models.npz')==receipt['models_sha256'] and digest(out/'steps.json')==receipt['steps_sha256']
    steps=json.loads((out/'steps.json').read_text());checked=0
    with np.load(out/'models.npz') as z:
        assert len(z['models'])==cfg['cases']
        for case,trajectory in enumerate(z['models']):
            R=z['relations'][case];keys=z['keys'];refs=z['refs'][case];center=trajectory[0]
            assert len(trajectory)==cfg['steps']+1
            for step,E in enumerate(trajectory):
                assert np.isfinite(E).all()
                for qi,(h,r) in enumerate(keys):
                    tree=E*(E[h]*R[r])
                    while tree.shape[1]>1:tree=tree[:,::2]+tree[:,1::2]
                    scores=tree[:,0]
                    for g in refs[refs[:,0]==qi,1]:assert 1+int((scores>scores[g]).sum())<=cfg['k'];checked+=1
                if step:
                    record=steps[case]['steps'][step-1];ids=record['ids'];unused=np.setdiff1d(np.arange(len(E)),ids)
                    assert np.array_equal(E[unused],trajectory[step-1][unused])
                    assert np.all(np.linalg.norm(E[ids]-center[ids],axis=1)<=np.asarray(record['radii'])+cfg['solver']['movement_tolerance'])
                    if not record['accepted']:assert np.array_equal(E,trajectory[step-1])
    result=dict(status='PASS_INDEPENDENT_ALL_SAVED_TRAJECTORY_RANKS_AND_BUDGETS',failures=[],member_ranks_checked=checked,
                receipt_sha256=digest(out/'receipt.json'),models_sha256=digest(out/'models.npz'),test_read=False,valid_read=False,
                limits=['Synthetic equivalence controls only; no complete real graph method or utility result.'])
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def aggregate(root,cfg):
    can=yaml.safe_load((root/'experiments'/cfg['id']/'experiment.yaml').read_text());assert not can['active_processes'];rows=[]
    for job in cfg['jobs']:
        if can['jobs'][job['id']]['status']=='completed_audited':
            out=root/cfg['reaudit_directory']/job['id'];rows.append(dict(result=json.loads((out/'receipt.json').read_text()),audit=json.loads((out/'audit.json').read_text())))
    dest=root/'results'/cfg['id'];dest.mkdir(exist_ok=False);(dest/'summary.json').write_text(json.dumps(dict(scope=cfg['scope'],jobs_passed=len(rows),jobs_total=len(cfg['jobs']),rows=rows),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job');p.add_argument('--stage',choices=['all','prepare','audit','aggregate'],default='all');a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));torch.set_num_threads(2);cfg=json.loads((root/a.config).read_text());job=next((j for j in cfg['jobs'] if j['id']==a.job),None)
    if a.stage=='all':
        for stage in ['prepare','audit']:subprocess.run([sys.executable,__file__,'--root',str(root),'--config',a.config,'--job',a.job,'--stage',stage],check=True)
    elif a.stage=='prepare':prepare(root,cfg,job)
    elif a.stage=='audit':audit(root,cfg,job)
    else:aggregate(root,cfg)
