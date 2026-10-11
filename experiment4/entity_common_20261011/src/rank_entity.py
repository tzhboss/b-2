"""Sequential native Rank updates on shared novel entities, with guarded QP.

Old embeddings/relations are frozen. Every new TRAIN fact is visited through its
new-tail direction(s); the same updated vector is used when an entity is a head.
This is a task adaptation, not the author's independent one-column benchmark.
"""
import importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from common_incremental import Data,metrics,ranks,evaluate
from common_training import sha


def native_core(root,cfg):
    path=(root/cfg['rank_source']).resolve()
    spec=importlib.util.spec_from_file_location('native_rank_novel_task',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def setup_embeddings(root,cfg,seed):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=root/'artifacts/base'/f'distmult-s{seed}'/'best.pt'
    audit=json.loads(base.with_name('audit.json').read_text());assert not audit.get('failures')
    b=torch.load(base,map_location='cuda:0',weights_only=True)
    gen=torch.Generator(device='cuda:0').manual_seed(seed+cfg['expansion_seed_offset'])
    init=torch.empty((11258,200),dtype=torch.float32,device='cuda:0')
    torch.nn.init.normal_(init,std=.12,generator=gen);init[:7505]=b['state_dict']['entity.weight']
    return init.double(),b['state_dict']['relation.weight'].double(),{str(base):sha(base)}


def cached(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}


@torch.no_grad()
def prepare_features(root,cfg,seed):
    started=time.monotonic();data=Data(root,cfg);old=data.triples(0,'train').numpy()
    E,R,bound=setup_embeddings(root,cfg,seed);core=native_core(root,cfg)
    folder=root/cfg['feature_directory']/f's{seed}';folder.mkdir(parents=True,exist_ok=False)
    selected=[];directions=[];gold=[];oldrank=[];perrel={}
    for off in range(0,len(old),cfg['feature_score_batch']):
        part=torch.as_tensor(old[off:off+cfg['feature_score_batch']],device=E.device)
        Q=E[part[:,0]]*R[part[:,1]];s=Q@E[:7505].T;g=s[torch.arange(len(part),device=E.device),part[:,2]]
        rank=1+(s>g[:,None]).sum(1)
        for q,v,t,rk in zip(part.cpu().numpy(),Q.cpu().numpy(),g.cpu().numpy(),rank.cpu().numpy()):
            rel=int(q[1])
            if rk<=10 and perrel.get(rel,0)<cfg['max_true_per_relation']:
                selected.append(q);directions.append(v);gold.append(t);oldrank.append(rk);perrel[rel]=perrel.get(rel,0)+1
    T=np.asarray(selected,dtype=np.int64).reshape(-1,3);G=np.asarray(directions,dtype=np.float64).reshape(-1,200)
    # Source ProtectionSubspace, same TRAIN order and cap5000 for each relation.
    basis=np.zeros((474,200,200),dtype=np.float64);brank=np.zeros(474,dtype=np.int64)
    for rel in range(474):
        sp=core.ProtectionSubspace.from_directions(G[T[:,1]==rel],dimension=200)
        basis[rel,:sp.rank]=sp.row_basis;brank[rel]=sp.rank
    queries=np.unique(old[:,:2],axis=0);P=[];cut=[];cutrank=[]
    for off in range(0,len(queries),cfg['feature_score_batch']):
        q=torch.as_tensor(queries[off:off+cfg['feature_score_batch']],device=E.device)
        Q=E[q[:,0]]*R[q[:,1]];s=Q@E[:7505].T;kth=torch.topk(s,10,dim=1).values[:,-1]
        P.append(Q.cpu().numpy());cut.extend(kth.cpu().numpy());cutrank.extend((1+(s>kth[:,None]).sum(1)).cpu().numpy())
    np.savez_compressed(folder/'guards.npz',true_queries=T,true_Q=G,true_gold=np.asarray(gold),true_old_rank=np.asarray(oldrank),
                        pred_queries=queries,pred_Q=np.concatenate(P),pred_cut=np.asarray(cut),pred_old_rank=np.asarray(cutrank),basis=basis,basis_rank=brank)
    bound.update(data.bound);bound[str(root/cfg['rank_source'])]=sha(root/cfg['rank_source'])
    receipt=dict(status='PASS_TRAIN_ONLY_GUARDS',seed=seed,test_read=False,new_train_facts_read=False,
                 true_guard_rows=len(T),prediction_query_rows=len(queries),elapsed_s=time.monotonic()-started,
                 bound_input_sha256=bound,output_sha256=sha(folder/'guards.npz'),implementation_sha256=sha(Path(__file__)),
                 precision='Exact float32 checkpoint/expansion values, float64 arithmetic',
                 scope='All relations; old7505 TRUE Top10 and old7505 predicted cutoff from old TRAIN only')
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2));print('RANK_FEATURES_READY',seed,len(T),len(queries),flush=True)


@torch.no_grad()
def guard_counts(Q,threshold,E,batch):
    out=[]
    for part,cut in zip(Q.split(batch),threshold.split(batch)):
        out.append((part@E.T>cut[:,None]).sum(1))
    return torch.cat(out)


def load_guards(root,cfg,seed,device):
    folder=root/cfg['feature_directory']/f's{seed}';receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['status']=='PASS_TRAIN_ONLY_GUARDS' and not receipt['test_read'] and not receipt['new_train_facts_read']
    assert sha(folder/'guards.npz')==receipt['output_sha256']
    for p,h in receipt['bound_input_sha256'].items():assert sha(p)==h
    f=cached(folder/'guards.npz');G=torch.as_tensor(f['true_Q'],device=device);P=torch.as_tensor(f['pred_Q'],device=device)
    return f,G,P,receipt


def subspace(core,f,rel):
    n=int(f['basis_rank'][rel]);directions=f['true_Q'][f['true_queries'][:,1]==rel]
    return core.ProtectionSubspace(directions,np.empty(n),f['basis'][rel,:n],n,200,1e-6)


@torch.no_grad()
def train_rank(root,cfg,job,preflight=False):
    started=time.monotonic();data=Data(root,cfg);old=data.triples(0,'train');new=data.triples(1,'train')
    E,R,bound=setup_embeddings(root,cfg,job['seed']);initial=E.clone();core=native_core(root,cfg)
    f,G,P,feature_receipt=load_guards(root,cfg,job['seed'],E.device)
    tg=torch.as_tensor(f['true_gold'],device=E.device);pg=torch.as_tensor(f['pred_cut'],device=E.device)
    tc=guard_counts(G,tg,E,cfg['feature_score_batch']);pc=guard_counts(P,pg,E,cfg['feature_score_batch'])
    pd=torch.as_tensor(f['pred_queries'],device=E.device)
    stage='rank-preflight' if preflight else 'incremental';out=root/'artifacts'/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    v=[data.triples(s,'valid') for s in (0,1)];vg=torch.Generator().manual_seed(cfg['validation_seed'])
    v=[q[torch.randperm(len(q),generator=vg)[:cfg['validation_queries_per_split']]] for q in v]
    order=torch.randperm(len(new),generator=torch.Generator().manual_seed(job['seed']+71)).numpy()
    eligible=order[new.numpy()[order,2]>=7505]
    assert set((eligible//2).tolist())==set(range(len(new)//2)), 'Each novel original TRAIN fact must be visited'
    if preflight:eligible=eligible[:cfg['preflight_updates']]
    # A truly new tail has no support in old TRAIN. This retains the original
    # Support-TRAIN least-squares objective, which reduces exactly to Direct.
    assert not (old[:,2]>=7505).any()
    sys.path.insert(0,str(root/cfg['qp_vendor']));from rank_constraints_v2 import solve_min_norm
    spaces={};records=[];replay_old=[];replay_new=[];replay_q=[];qp_delta=[];rank_delta=[];count0_t=tc.clone();count0_p=pc.clone()
    replay_ids=[];best=-1.;epochs=[];best_epoch=0;certified=0;fallback=0;opposing=0;selfloops=0
    for epoch in range(1,1+cfg['sweeps'] if not preflight else 2):
        for i,index in enumerate(eligible):
            assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Registered novel Rank/QP wall-clock budget exceeded'
            h,rel,t=map(int,new[index]);q=(E[h]*R[rel]).cpu().numpy();s=(E[h]*R[rel]@E.T).cpu().numpy()
            gap=core.promotion_score_gap(s,t,top_k=10,margin=cfg['promotion_margin'])
            if rel not in spaces:spaces[rel]=subspace(core,f,rel)
            sp=spaces[rel];rd=core.rank_truncated_delta(q,gap,sp,cfg['rank_fraction'])[0]
            method=job['method'];d=rd.copy();certificate=None;used=False;cap=float(np.linalg.norm(rd));ct0=tc;cp0=pc
            before=E[t].clone();tscore=G@before;pscore=P@before
            true_mask=(tc==9)&(tscore<=tg)
            pred_mask=(pc==9)&(pscore<=pg)&~((pd[:,0]==h)&(pd[:,1]==rel))
            if method=='DirectNovel':d=core.direct_promotion_delta(q,gap)
            elif method=='SupportTrainNovel':d=np.linalg.lstsq(q[None,:],np.asarray([gap]),rcond=None)[0]
            elif method=='StrictNullNovel':d=core.rank_truncated_delta(q,gap,sp,1.)[0]
            elif method=='RidgeNovel':d=core.ridge_preservation_delta(q,gap,sp,cfg['ridge_lambda'])
            elif method in ['QPTrueRankNovel','QPPredRankNovel'] and gap>0:
                # All guard heads/tails are old, so their scores are affine in
                # this one new entity even though the global model is shared.
                A=[q[None,:]];b=[np.asarray([gap])]
                A.append(-G[true_mask].cpu().numpy());b.append((tscore[true_mask]-tg[true_mask]+cfg['guard_margin']).cpu().numpy())
                if method=='QPPredRankNovel':
                    A.append(-P[pred_mask].cpu().numpy());b.append((pscore[pred_mask]-pg[pred_mask]+cfg['guard_margin']).cpu().numpy())
                A=np.vstack(A);b=np.concatenate(b);conflict=np.any(np.all(A[1:]==-q,axis=1)&(gap>-b[1:]))
                if conflict:opposing+=1;certificate=dict(certified=False,status='exact_opposing_promotion_guard_conflict')
                else:
                    redundant=b < -np.linalg.norm(A,axis=1)*cap-1e-10;redundant[0]=False
                    candidate,certificate=solve_min_norm(A[~redundant],b[~redundant],maxiter=cfg['qp_maxiter'])
                    certificate['full_system_rows']=len(A);certificate['pruned_rows']=int(redundant.sum())
                    certificate['full_primal_min']=float((A@candidate-b).min())
                    if certificate['certified'] and certificate['full_primal_min']>=-cfg['qp_primal_tolerance'] and np.linalg.norm(candidate)<=cap+cfg['norm_cap_tolerance']:
                        d=candidate;used=True
            proposed=before+torch.as_tensor(d,device=E.device)
            tnew=G@proposed;pnew=P@proposed
            next_tc=tc+(tnew>tg).long()-(tscore>tg).long();next_pc=pc+(pnew>pg).long()-(pscore>pg).long()
            E[t]=proposed
            after_s=(E[h]*R[rel]@E.T);actual_rank=int(1+(after_s>after_s[t]).sum())
            true_losses=int(((tc<=9)&(next_tc>9)).sum());pred_losses=int(((pc<=9)&(next_pc>9)&~((pd[:,0]==h)&(pd[:,1]==rel))).sum())
            if used and (actual_rank>10 or true_losses or (method=='QPPredRankNovel' and pred_losses)):
                used=False;d=rd.copy();E[t]=before+torch.as_tensor(d,device=E.device)
                tnew=G@E[t];pnew=P@E[t]
                next_tc=tc+(tnew>tg).long()-(tscore>tg).long();next_pc=pc+(pnew>pg).long()-(pscore>pg).long()
                after_s=(E[h]*R[rel]@E.T);actual_rank=int(1+(after_s>after_s[t]).sum())
                certificate['rejected_actual_shared_model_acceptance']=True
            if used:certified+=1
            elif method.startswith('QP') and gap>0:fallback+=1
            if h==t:selfloops+=1
            replay_old.append(before.cpu().numpy());replay_new.append(E[t].cpu().numpy());replay_q.append(q);qp_delta.append(d);rank_delta.append(rd);replay_ids.append([epoch,int(index),h,rel,t])
            records.append(dict(epoch=epoch,directed_index=int(index),h=h,r=rel,t=t,gap=gap,rank_after=actual_rank,
                                rank_norm_cap=cap,qp_used=used,certificate=certificate,
                                true_boundary_rows=int(true_mask.sum()),pred_boundary_rows=int(pred_mask.sum()),selfloop=h==t))
            tc=next_tc;pc=next_pc
            if (i+1)%cfg['progress_every_updates']==0:
                print('RANK_NOVEL_PROGRESS',job['id'],epoch,i+1,len(eligible),round(time.monotonic()-started,2),certified,fallback,flush=True)
        values=[metrics(ranks(E,R,q[:16] if preflight else q,'DistMult')) for q in v]
        criterion=.5*(values[0]['raw_mrr']+values[1]['raw_mrr'])
        epochs.append(dict(epoch=epoch,validation=values,selection_score=criterion,elapsed_s=time.monotonic()-started))
        if criterion>best:
            best=criterion;best_epoch=epoch;torch.save(dict(E=E.cpu(),R=R.cpu(),kind='DistMult',seed=job['seed'],epoch=epoch,method=method),out/'best.pt')
        print('RANK_NOVEL_EPOCH',job['id'],epoch,criterion,flush=True)
    # Full TRAIN guard recount certifies tracking at the terminal state, no TEST.
    fresh_t=guard_counts(G,tg,E,cfg['feature_score_batch']);fresh_p=guard_counts(P,pg,E,cfg['feature_score_batch'])
    assert torch.equal(fresh_t,tc) and torch.equal(fresh_p,pc),'Incremental guard tracking differs from full candidate recount'
    assert torch.equal(E[:7505],initial[:7505])
    bound.update(data.bound);bound[str(root/cfg['rank_source'])]=sha(root/cfg['rank_source'])
    for path in (root/cfg['qp_vendor']).glob('*.py'):bound[str(path)]=sha(path)
    bound[str(root/cfg['feature_directory']/f"s{job['seed']}"/'receipt.json')]=sha(root/cfg['feature_directory']/f"s{job['seed']}"/'receipt.json')
    for p,h in bound.items():assert sha(p)==h
    torch.save(dict(E=initial.cpu(),R=R.cpu(),oldE=initial[:7505].cpu()),out/'initial.pt')
    np.savez_compressed(out/'training-replay.npz',ids=np.asarray(replay_ids),old_vectors=np.asarray(replay_old),new_vectors=np.asarray(replay_new),
                        q=np.asarray(replay_q),actual_delta=np.asarray(qp_delta),rank_delta=np.asarray(rank_delta),
                        initial_true_counts=count0_t.cpu().numpy(),initial_pred_counts=count0_p.cpu().numpy(),
                        terminal_true_counts=tc.cpu().numpy(),terminal_pred_counts=pc.cpu().numpy())
    (out/'steps.json').write_text(json.dumps(records));(out/'training.json').write_text(json.dumps(epochs,indent=2))
    receipt=dict(id=job['id'],method=method,kind='DistMult',seed=job['seed'],stage=stage,test_read=False,best_epoch=best_epoch,best_validation_score=best,
                 epochs_executed=len(epochs),elapsed_s=time.monotonic()-started,parameters=11258*200+474*200,trainable_parameters=3753*200,
                 peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),bound_input_sha256=bound,checkpoint_sha256=sha(out/'best.pt'),
                 novel_original_facts=len(new)//2,eligible_directed_facts=int((new[:,2]>=7505).sum()),processed_updates=len(records),
                 all_original_facts_visited=not preflight,qp_accepted=certified,qp_fallbacks=fallback,exact_opposing_conflicts=opposing,selfloops=selfloops,
                 feature_preparation_seconds=feature_receipt['elapsed_s'],old_parameters_bitwise_unchanged=True,shared_head_tail=True,
                 terminal_guard_recount_exact=True,implementation_sha256=sha(Path(__file__)),
                 limits=['Sequential shared-new-entity task adaptation, not native independent one-column protocol.',
                         'TRUE and Prediction acceptance protects the currently retained TRAIN frontier; excludes the edited Prediction query.',
                         'Initial expansion losses, fallback steps and unprotected TEST queries have no hard guarantee.',
                         'Float64 scoring of the same float32 checkpoint values; one full sweep, not equal optimization to 200-epoch SGD.'])
    (out/'training-receipt.json').write_text(json.dumps(receipt,indent=2));print('RANK_NOVEL_TERMINAL',job['id'],flush=True)


def audit_rank_steps(root,cfg,job):
    """Replay every accepted QP's original full linear system independently."""
    from scipy.optimize import nnls
    torch.set_num_threads(2);out=root/'artifacts/incremental'/job['id'];t=json.loads((out/'training-receipt.json').read_text())
    f=cached(root/cfg['feature_directory']/f"s{job['seed']}"/'guards.npz');z=cached(out/'training-replay.npz')
    G=f['true_Q'];P=f['pred_Q'];tg=f['true_gold'];pg=f['pred_cut'];pq=f['pred_queries']
    tc=z['initial_true_counts'].copy();pc=z['initial_pred_counts'].copy();rows=json.loads((out/'steps.json').read_text());checks=0;accepted=0;failures=[];maximum_violation=0.;worst_kkt=0.
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True);E=initial['E'].numpy().copy();R=initial['R'].numpy()
    for i,row in enumerate(rows):
        epoch,index,h,rel,tail=map(int,z['ids'][i]);old=z['old_vectors'][i];new=z['new_vectors'][i];d=new-old
        assert np.array_equal(E[tail],old),'Shared entity replay continuity'
        assert np.array_equal(z['q'][i],E[h]*R[rel]),'Updated head participates in true scoring'
        oldt=G@old;newt=G@new;oldp=P@old;newp=P@new
        mask=(tc==9)&(oldt<=tg);pmask=(pc==9)&(oldp<=pg)&~((pq[:,0]==h)&(pq[:,1]==rel))
        nt=tc+(newt>tg).astype(np.int64)-(oldt>tg).astype(np.int64);np_=pc+(newp>pg).astype(np.int64)-(oldp>pg).astype(np.int64)
        E[tail]=new;checks+=3
        if row['qp_used']:
            accepted+=1;A=[z['q'][i][None,:],-G[mask]];b=[np.array([row['gap']]),oldt[mask]-tg[mask]+cfg['guard_margin']]
            if job['method']=='QPPredRankNovel':A.append(-P[pmask]);b.append(oldp[pmask]-pg[pmask]+cfg['guard_margin'])
            A=np.vstack(A);b=np.concatenate(b);slack=A@d-b
            violation=max(0.,float(-slack.min()));maximum_violation=max(maximum_violation,violation)
            assert violation<=cfg['qp_primal_tolerance'] and np.linalg.norm(d)<=row['rank_norm_cap']+cfg['norm_cap_tolerance']
            assert not np.any((tc<=9)&(nt>9))
            if job['method']=='QPPredRankNovel':assert not np.any((pc<=9)&(np_>9)&~((pq[:,0]==h)&(pq[:,1]==rel)))
            score=(E[h]*R[rel])@E.T;assert 1+np.count_nonzero(score>score[tail])<=10
            # The native solver reports full KKT diagnostics. Reconstruct an
            # independent nonnegative dual witness from the accepted vector.
            lengths=np.linalg.norm(A,axis=1);nz=lengths>0;a=A[nz]/lengths[nz,None];rhs=b[nz]/lengths[nz]
            active=np.abs(a@d-rhs)<=cfg['audit_active_tolerance']
            if active.any():
                dual,residual=nnls(a[active].T,d,maxiter=cfg['audit_nnls_maxiter']);stat=float(np.max(np.abs(a[active].T@dual-d)))
                complement=float(np.max(np.abs(dual*(a[active]@d-rhs))))
                assert stat<=cfg['audit_kkt_tolerance'] and complement<=cfg['audit_kkt_tolerance']
                worst_kkt=max(worst_kkt,stat,complement)
            else:assert np.linalg.norm(d)<=cfg['audit_kkt_tolerance']
            checks+=6
        elif job['method'].startswith('QP'):assert np.array_equal(z['actual_delta'][i],z['rank_delta'][i]);checks+=1
        tc=nt;pc=np_
        if (i+1)%2048==0:print('QP_REPLAY_AUDIT',job['id'],i+1,len(rows),accepted,flush=True)
    assert np.array_equal(tc,z['terminal_true_counts']) and np.array_equal(pc,z['terminal_pred_counts'])
    assert accepted==t['qp_accepted'];assert t['all_original_facts_visited']
    # Every replayed original fact ID must be covered, not just easy edit rows.
    assert set((z['ids'][:,1]//2).tolist())==set(range(t['novel_original_facts']))
    replay_state=dict(status='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS',checks=checks,accepted_qp_checked=accepted,
                      steps_replayed=len(rows),max_original_primal_violation=maximum_violation,worst_independent_kkt_error=worst_kkt,
                      scope='All steps continuity/guard crossing; every accepted QP full primal/cap/target/frontier and reconstructed nonnegative dual',
                      implementation_sha256=sha(Path(__file__)))
    (out/'solver-audit.json').write_text(json.dumps(replay_state,indent=2));print(json.dumps(replay_state))
