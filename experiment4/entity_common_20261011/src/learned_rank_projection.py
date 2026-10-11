"""Learned shared-vector proposals, optional SVD and rank-frontier projection.

Known components, explicit common-task adaptation. No trained-graph run before
registered comparison dependencies finish. Existing producer code is reused
without mutation; all new output namespaces are disjoint.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from common_incremental import Data,metrics,ranks,evaluate
from common_training import sha
from rank_entity import setup_embeddings,load_guards,guard_counts,cached
from safe_proposal import nearest_feasible,radius_segment,verify_projection


def prepare_basis(root,cfg,seed):
    torch.set_num_threads(2);folder=root/cfg['feature_directory']/f's{seed}'
    receipt=json.loads((folder/'receipt.json').read_text());assert receipt['status']=='PASS_TRAIN_ONLY_GUARDS' and not receipt['test_read']
    assert sha(folder/'guards.npz')==receipt['output_sha256'];f=cached(folder/'guards.npz');G=f['true_Q']
    gram=G.T@G;eigenvalues,vectors=np.linalg.eigh(gram);rank=int((eigenvalues>max(eigenvalues[-1],0)*cfg['basis_rcond']**2).sum())
    keep=int(np.ceil(cfg['rank_fraction']*rank));basis=vectors[:,-keep:].T.copy() if keep else np.empty((0,200))
    assert np.allclose(basis@basis.T,np.eye(keep),atol=1e-12,rtol=0)
    out=root/cfg['basis_directory']/f's{seed}';out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'basis.npz',basis=basis,eigenvalues=eigenvalues,rank=rank)
    r=dict(check_scope='Old TRAIN all-relation TRUE query eigenspace only',test_read=False,new_train_read=False,rank=rank,removed_directions=keep,
           guard_receipt_sha256=sha(folder/'receipt.json'),guard_arrays_sha256=sha(folder/'guards.npz'),basis_sha256=sha(out/'basis.npz'))
    (out/'receipt.json').write_text(json.dumps(r,indent=2));print('LEARNED_PROPOSAL_BASIS',seed,rank,keep,flush=True)


def load_basis(root,cfg,seed):
    out=root/cfg['basis_directory']/f's{seed}';r=json.loads((out/'receipt.json').read_text());assert not r['test_read'] and not r['new_train_read']
    assert r['basis_sha256']==sha(out/'basis.npz') and r['guard_arrays_sha256']==sha(root/cfg['feature_directory']/f's{seed}'/'guards.npz')
    return cached(out/'basis.npz')['basis'],r


def source_proposal(root,cfg,job,initial,R):
    out=root/'artifacts/incremental'/cfg['proposal_job_template'].format(seed=job['seed']);audit=json.loads((out/'audit.json').read_text());assert audit['status'].startswith('PASS') and not audit.get('failures')
    receipt=json.loads((out/'training-receipt.json').read_text());assert not receipt['test_read'] and receipt['method']=='NewEntityOnly' and receipt['kind']=='DistMult'
    assert receipt['checkpoint_sha256']==sha(out/'best.pt')
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True);E=checkpoint['E'].double();rr=checkpoint['R'].double()
    init=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
    assert E.shape==(11258,200) and rr.shape==(474,200) and torch.isfinite(E).all() and torch.isfinite(rr).all()
    assert torch.equal(E[:7505],initial[:7505].cpu()) and torch.equal(rr,R.cpu())
    assert torch.equal(init['E'].double(),initial.cpu()) and torch.equal(init['R'].double(),R.cpu())
    bound={str(out/name):sha(out/name) for name in ['best.pt','initial.pt','audit.json','training-receipt.json']}
    for name,digest in receipt['bound_input_sha256'].items():assert sha(name)==digest
    bound.update(receipt['bound_input_sha256']);return E.to(initial.device),receipt,bound


def frontier_cpu(G,P,tg,pg,old,tc,pc,good_t,good_p,protection,margin):
    s=G@old;p=P@old;tm=good_t&(tc==9)&(s<=tg);pm=good_p&(pc==9)&(p<=pg)
    matrices=[];bounds=[]
    if protection!='none':matrices.append(G[tm]);bounds.append(tg[tm]-s[tm])
    if protection=='pred':matrices.append(P[pm]);bounds.append(pg[pm]-p[pm])
    A=np.concatenate(matrices) if matrices else np.empty((0,len(old)));b=np.concatenate(bounds) if bounds else np.empty(0)
    assert (b>=0).all();b=b-np.minimum(margin,.5*b)
    return A,b


@torch.no_grad()
def train_rank(root,cfg,job,preflight=False):
    started=time.monotonic();E,R,bound=setup_embeddings(root,cfg,job['seed']);initial=E.clone()
    radius=float(torch.linalg.vector_norm(initial,dim=1).max().cpu())
    target,source,source_bound=source_proposal(root,cfg,job,initial,R);bound.update(source_bound)
    f,G,P,guard_receipt=load_guards(root,cfg,job['seed'],E.device);basis,basis_receipt=load_basis(root,cfg,job['seed'])
    bound[str(root/cfg['feature_directory']/f"s{job['seed']}"/'receipt.json')]=sha(root/cfg['feature_directory']/f"s{job['seed']}"/'receipt.json')
    bound[str(root/cfg['basis_directory']/f"s{job['seed']}"/'receipt.json')]=sha(root/cfg['basis_directory']/f"s{job['seed']}"/'receipt.json')
    tg=torch.as_tensor(f['true_gold'],device=E.device);pg=torch.as_tensor(f['pred_cut'],device=E.device)
    tc=guard_counts(G,tg,E,cfg['feature_score_batch']);pc=guard_counts(P,pg,E,cfg['feature_score_batch']);count0_t=tc.clone();count0_p=pc.clone();good_t=tc<=9;good_p=pc<=9
    stage='proposal-preflight' if preflight else 'incremental';out=root/'artifacts'/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    order=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy()
    if preflight:order=order[:cfg['preflight_updates']]
    before=[];after=[];proposals=[];projected=[];steps=[];zero_count=0;certified_count=0;rejected_count=0
    for index,tail in enumerate(order):
        assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Registered projection budget exceeded'
        tail=int(tail);old=E[tail].cpu().numpy().copy();u=(target[tail]-E[tail]).cpu().numpy()
        if job['apply_rank']:u=u-basis.T@(basis@u)
        st=G@E[tail];sp=P@E[tail]
        tm=good_t&(tc==9)&(st<=tg);pm=good_p&(pc==9)&(sp<=pg)
        aa=[];bb=[]
        if job['protection']!='none':aa.append(G[tm]);bb.append(tg[tm]-st[tm])
        if job['protection']=='pred':aa.append(P[pm]);bb.append(pg[pm]-sp[pm])
        A=torch.cat(aa).cpu().numpy() if aa else np.empty((0,200));b=torch.cat(bb).cpu().numpy() if bb else np.empty(0)
        b=b-np.minimum(cfg['guard_margin'],.5*b);assert (b>=0).all()
        if job['protection']!='none':d,certificate=nearest_feasible(u,A,b,cfg['solver'])
        else:d=u.copy();certificate=dict(certified=False,status='unprotected_control',dual_rows=[],dual=[])
        alpha=radius_segment(old,d,radius) if job['apply_rank'] or job['protection']!='none' else 1.
        candidate=E[tail]+torch.as_tensor(alpha*d,device=E.device);nt=tc+(G@candidate>tg).long()-(st>tg).long();np_=pc+(P@candidate>pg).long()-(sp>pg).long()
        reject=job['protection']!='none' and (bool((good_t&(nt>9)).any()) or (job['protection']=='pred' and bool((good_p&(np_>9)).any())))
        if reject:candidate=E[tail].clone();nt=tc;np_=pc;rejected_count+=1
        actual=candidate.cpu().numpy().copy();assert np.isfinite(actual).all()
        if np.array_equal(old,actual):zero_count+=1
        if job['protection']!='none' and certificate['certified']:certified_count+=1
        E[tail]=candidate;before.append(old);after.append(actual);proposals.append(u);projected.append(d)
        steps.append(dict(entity=tail,alpha=alpha,actual_guard_rejection=bool(reject),certificate=certificate,
                          frontier_true=int(tm.sum()),frontier_pred=int(pm.sum()),radius=radius))
        tc=nt;pc=np_
        if (index+1)%cfg['progress_every_updates']==0:print('LEARNED_PROPOSAL_PROGRESS',job['id'],index+1,len(order),zero_count,certified_count,flush=True)
    assert torch.equal(guard_counts(G,tg,E,cfg['feature_score_batch']),tc) and torch.equal(guard_counts(P,pg,E,cfg['feature_score_batch']),pc),'Full terminal guard tracking mismatch'
    assert torch.equal(E[:7505],initial[:7505])
    if job['protection']!='none':assert not (good_t&(tc>9)).any()
    if job['protection']=='pred':assert not (good_p&(pc>9)).any()
    if job['protection']=='none' and not job['apply_rank']:
        # Addition can round differently from assigning the target. Source-only
        # control must be bitwise target, not a silently changed geometry.
        assert torch.allclose(E[order],target[order],atol=1e-15,rtol=0)
        E[order]=target[order]
        # Recount is explicit after exact source assignment.
        tc=guard_counts(G,tg,E,cfg['feature_score_batch']);pc=guard_counts(P,pg,E,cfg['feature_score_batch'])
        after=[E[int(tail)].cpu().numpy().copy() for tail in order]
    data=Data(root,cfg);valid=[data.triples(s,'valid') for s in (0,1)];vg=torch.Generator().manual_seed(cfg['validation_seed']);valid=[q[torch.randperm(len(q),generator=vg)[:cfg['validation_queries_per_split']]] for q in valid]
    source_values=[metrics(ranks(target,R,q,'DistMult')) for q in valid];values=[metrics(ranks(E,R,q,'DistMult')) for q in valid]
    old_before=ranks(initial[:7505],R,valid[0],'DistMult');old_after=ranks(E,R,valid[0],'DistMult');source_old=ranks(target,R,valid[0],'DistMult');protected=old_before<=10
    validation=dict(source=source_values,after=values,source_old_top10_loss=float((source_old[protected]>10).mean()),after_old_top10_loss=float((old_after[protected]>10).mean()))
    torch.save(dict(E=E.cpu(),R=R.cpu(),kind='DistMult',method=job['method'],seed=job['seed'],epoch=1),out/'best.pt')
    torch.save(dict(E=initial.cpu(),R=R.cpu(),oldE=initial[:7505].cpu()),out/'initial.pt')
    np.savez_compressed(out/'projection-replay.npz',entity_ids=order,before=np.asarray(before),after=np.asarray(after),proposal=np.asarray(proposals),projected=np.asarray(projected),initial_true_counts=count0_t.cpu().numpy(),initial_pred_counts=count0_p.cpu().numpy(),terminal_true_counts=tc.cpu().numpy(),terminal_pred_counts=pc.cpu().numpy())
    (out/'steps.json').write_text(json.dumps(steps));(out/'training.json').write_text(json.dumps([dict(epoch=1,validation=validation)]))
    bound.update(data.bound)
    for name,digest in bound.items():assert sha(name)==digest
    result=dict(id=job['id'],method=job['method'],kind='DistMult',seed=job['seed'],test_read=False,preflight=preflight,epochs_executed=1,
                elapsed_s=time.monotonic()-started,source_sgd_elapsed_s=source['elapsed_s'],processed_new_entities=len(order),zero_updates=zero_count,certified_projections=certified_count,
                actual_guard_rejections=rejected_count,source_checkpoint_selection='Reuse audited NewEntityOnly common VALID selected checkpoint; exposed development source',
                bound_input_sha256=bound,checkpoint_sha256=sha(out/'best.pt'),implementation_sha256=sha(Path(__file__)),validation=validation,
                projection_replay_sha256=sha(out/'projection-replay.npz'),steps_sha256=sha(out/'steps.json'),entity_radius=radius,
                initially_retained_true=int(good_t.sum()),initially_retained_prediction=int(good_p.sum()),terminal_guard_recount_exact=True,
                parameters=11258*200+474*200,trainable_parameters=0,modifiable_parameters=3753*200,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                limits=['Known learned-proposal/SVD/Euclidean projection combination, not confirmed algorithmic novelty.',
                        'Strict guard cutoff/count contract on selected old TRAIN initially retained members only; no all-TEST rank guarantee.',
                        'Projection failures yield zero, and useful learning requires the registered utility screen.',
                        'SVD uses all-relation TRUE directions and trained proposals, not native analytic single-edit Rank formula.',
                        'Original source SGD, basis preparation and postprocessing costs must be reported separately.'])
    (out/'training-receipt.json').write_text(json.dumps(result,indent=2));print('LEARNED_PROPOSAL_TERMINAL',job['id'],flush=True)


def audit_rank_steps(root,cfg,job):
    torch.set_num_threads(2);out=root/'artifacts/incremental'/job['id'];tr=json.loads((out/'training-receipt.json').read_text());assert not tr['test_read'] and not tr['preflight']
    for name,digest in tr['bound_input_sha256'].items():assert sha(name)==digest
    assert sha(out/'projection-replay.npz')==tr['projection_replay_sha256'] and sha(out/'steps.json')==tr['steps_sha256']
    assert sha(out/'best.pt')==tr['checkpoint_sha256'];z=cached(out/'projection-replay.npz');steps=json.loads((out/'steps.json').read_text())
    assert len(steps)==3753 and np.array_equal(np.sort(z['entity_ids']),np.arange(7505,11258))
    expected=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy();assert np.array_equal(expected,z['entity_ids'])
    init=torch.load(out/'initial.pt',map_location='cpu',weights_only=True);E=init['E'].numpy().copy();R=init['R'].numpy()
    target,_,_=source_proposal(root,cfg,job,init['E'],init['R']);target=target.numpy();basis,_=load_basis(root,cfg,job['seed'])
    f=cached(root/cfg['feature_directory']/f"s{job['seed']}"/'guards.npz');G=f['true_Q'];P=f['pred_Q'];tg=f['true_gold'];pg=f['pred_cut']
    tc=z['initial_true_counts'].copy();pc=z['initial_pred_counts'].copy();good_t=tc<=9;good_p=pc<=9;radius=float(np.linalg.norm(E,axis=1).max());checks=0;zero=0;certified=0
    for index,row in enumerate(steps):
        tail=int(z['entity_ids'][index]);old=z['before'][index];new=z['after'][index];assert row['entity']==tail and np.array_equal(E[tail],old)
        u=target[tail]-old
        if job['apply_rank']:u=u-basis.T@(basis@u)
        assert np.allclose(u,z['proposal'][index],atol=1e-13,rtol=0)
        A,b=frontier_cpu(G,P,tg,pg,old,tc,pc,good_t,good_p,job['protection'],cfg['guard_margin'])
        d=z['projected'][index]
        if job['protection']!='none':verify_projection(u,A,b,d,row['certificate'],cfg['solver']);certified+=int(row['certificate']['certified'])
        else:assert np.array_equal(d,z['proposal'][index])
        alpha=radius_segment(old,d,radius) if job['apply_rank'] or job['protection']!='none' else 1.
        assert abs(alpha-row['alpha'])<=1e-14
        if row['actual_guard_rejection']:assert np.array_equal(new,old)
        else:assert np.allclose(new,old+alpha*d,atol=1e-13,rtol=0)
        st=G@old;sp=P@old;nt=tc+(G@new>tg).astype(np.int64)-(st>tg).astype(np.int64);np_=pc+(P@new>pg).astype(np.int64)-(sp>pg).astype(np.int64)
        if job['protection']!='none':assert not (good_t&(nt>9)).any()
        if job['protection']=='pred':assert not (good_p&(np_>9)).any()
        if job['apply_rank'] or job['protection']!='none':assert np.linalg.norm(new)<=radius+cfg['solver']['norm_tolerance']
        zero+=int(np.array_equal(new,old));E[tail]=new;tc=nt;pc=np_;checks+=8
        if (index+1)%512==0:print('LEARNED_PROPOSAL_AUDIT',job['id'],index+1,3753,flush=True)
    assert np.array_equal(tc,z['terminal_true_counts']) and np.array_equal(pc,z['terminal_pred_counts'])
    final=torch.load(out/'best.pt',map_location='cpu',weights_only=True);assert np.array_equal(E,final['E'].numpy()) and np.array_equal(R,final['R'].numpy())
    assert np.array_equal(E[:7505],init['oldE'].numpy()) and zero==tr['zero_updates'] and certified==tr['certified_projections']
    result=dict(status='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS',checks=checks,steps_replayed=3753,certified_projections=certified,zero_updates=zero,
                failures=[],scope='All3753 learned vector steps, full frontier systems, sparse nonnegative dual witnesses, radius, guard crossings and frozen oldE/R',
                limits=['No independent source SGD retraining.','Registered fixed TRAIN cutoff/count scope; full common TEST scoring remains a separate audit.'],implementation_sha256=sha(Path(__file__)))
    (out/'solver-audit.json').write_text(json.dumps(result,indent=2));print('LEARNED_PROPOSAL_ALL_STEP_AUDIT_PASS',job['id'],flush=True)
