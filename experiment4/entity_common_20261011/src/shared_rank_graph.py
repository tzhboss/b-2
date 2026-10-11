"""Common graph joint/shared-row updates and TRAIN-only parameter selection."""
import json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from common_incremental import Data,evaluate,metrics,ranks
from common_training import sha
from shared_rank_scope import source_initial
from shared_rank_cache_v2 import SharedRankCache
from shared_rank_solver_v2 import solve_shared_update


def load_scope(root,cfg,seed,mode):
    folder=root/cfg['scope_directory']/f'shared-rank-scope-s{seed}'
    audit=json.loads((folder/'audit.json').read_text());assert audit['status'].startswith('PASS') and not audit['failures']
    receipt=json.loads((folder/'receipt.json').read_text())
    for name,digest in receipt['input_sha256'].items():assert sha(Path(name))==digest
    assert receipt['output_sha256']['scope.npz']==sha(folder/'scope.npz')
    with np.load(folder/'scope.npz') as z:keys=z[mode+'_keys'].copy();refs=z[mode+'_refs'].copy()
    return keys,refs,{str(p):sha(p) for p in [folder/'scope.npz',folder/'receipt.json',folder/'audit.json']}


def choose_old_budget(data,cache,target,initial,options):
    """All selection depends on new/old TRAIN and current initial scores only."""
    new=data.triples(1,'train').numpy();old=data.triples(0,'train').numpy()
    positive=defaultdict(set);degree=np.zeros(7505,dtype=np.int64)
    for h,r,t in old:positive[(int(h),int(r))].add(int(t));degree[int(h)]+=1
    query_index={tuple(k):i for i,k in enumerate(cache.keys.cpu().numpy())}
    E=initial.cpu().numpy();R=cache.R.cpu().numpy();desired=target.cpu().numpy()
    voters=defaultdict(list);entity_queries=defaultdict(list)
    for h,r,t in new:
        if h<7505 and t>=7505 and (int(h),int(r)) in query_index:entity_queries[int(t)].append((int(h),int(r)))
    for entity,queries in entity_queries.items():
        for h,r in sorted(set(queries)):
            qi=query_index[(h,r)];s=cache.S[qi,:7505].cpu().numpy();boundary=float(cache.thresholds[qi])
            new_score=float(np.dot(E[h]*R[r],desired[entity]))
            if cache.counts[qi].item()!=9 or new_score<=boundary:continue
            candidates=[int(c) for c in np.flatnonzero(s>boundary) if int(c) not in positive[(h,r)] and int(c)!=h]
            if not candidates:continue
            # Closest old competitor above the boundary; important high-degree
            # vectors get a larger movement penalty, not an undeclared label.
            candidate=min(candidates,key=lambda c:(s[c]-boundary,degree[c],c))
            voters[candidate].append(entity)
    selected=sorted(voters,key=lambda c:(-len(set(voters[c])),degree[c],c))[:options['max_old_rows']]
    eligible=set(selected);choices={}
    for candidate,entities in voters.items():
        if candidate in eligible:
            for entity in entities:choices.setdefault(entity,[]).append(candidate)
    per_entity={int(e):min(set(candidates),key=lambda c:(degree[c],c)) for e,candidates in choices.items()}
    return selected,per_entity,degree,dict(selection='TRAIN new-answer rank10 barriers; nearest unlabelled old competitor; votes then degree/ID; one old row per new entity',
                                         max_old_rows=options['max_old_rows'],selected_old_rows=selected,new_entities_with_old_candidate=len(per_entity),
                                         unlabelled_is_not_false_knowledge=True)


@torch.no_grad()
def train_rank(root,cfg,job,preflight=False):
    started=time.monotonic();torch.set_num_threads(2);device=torch.device('cuda:0')
    data=Data(root,cfg);initial,R,bound=source_initial(root,cfg,job['seed'],device)
    source=root/'artifacts/incremental'/cfg['source_template'].format(seed=job['seed'])
    source_receipt=json.loads((source/'training-receipt.json').read_text());assert not source_receipt['test_read']
    assert sha(source/'best.pt')==source_receipt['checkpoint_sha256']
    target_model=torch.load(source/'best.pt',map_location=device,weights_only=True);target=target_model['E'].double()
    assert torch.equal(target[:7505],initial[:7505]) and torch.equal(target_model['R'].double(),R)
    bound.update({str(p):sha(p) for p in [source/'best.pt',source/'training-receipt.json']})
    keys,refs,scope_bound=load_scope(root,cfg,job['seed'],job['protection_scope']);bound.update(scope_bound)
    cache=SharedRankCache(initial,R,keys,refs,cfg['k'],cfg['score_batch'])
    selected,per_entity,degree,selection=choose_old_budget(data,cache,target,initial,cfg['parameter_budget']) if job['joint_old'] else ([],{},np.zeros(7505),dict(selected_old_rows=[],new_entities_with_old_candidate=0))
    order=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy()
    if preflight:
        chosen=[int(e) for e in order if int(e) in per_entity][:cfg['preflight_joint_updates']]
        remainder=[int(e) for e in order if int(e) not in set(chosen)]
        order=np.asarray(chosen+remainder[:cfg['preflight_updates']-len(chosen)],dtype=np.int64)
    stage='shared-preflight' if preflight else 'incremental';out=root/'artifacts'/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    steps=[];before=[];after=[];ids_records=[];accepted=0;fallback=0;old_changes=0
    radius=float(torch.linalg.vector_norm(initial,dim=1).max().cpu());median_degree=float(np.median(degree[degree>0])) if selected else 1.
    for index,entity in enumerate(order):
        assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Registered shared graph budget exceeded'
        entity=int(entity);old_id=per_entity.get(entity);ids=[entity] if old_id is None else [old_id,entity]
        rows=cache.E[ids].cpu().numpy().copy();desired=rows.copy();desired[-1]=target[entity].cpu().numpy()
        centers=initial[ids].cpu().numpy().copy();weights=[1.] if old_id is None else [cfg['parameter_budget']['old_penalty']*(1+degree[old_id]/median_degree),1.]
        radii=[2*radius] if old_id is None else [cfg['parameter_budget']['old_radius_fraction']*float(np.linalg.norm(centers[0])),2*radius]
        if old_id is not None:desired[0]=centers[0]
        patch,receipt=solve_shared_update(cache,ids,desired,weights,centers,radii,[] if old_id is None else [old_id],cfg['solver'])
        candidate=patch.rows.cpu().numpy().copy()
        if receipt['accepted']:
            assert cache.commit(patch);accepted+=1
        else:
            assert np.array_equal(candidate,rows);fallback+=1
        if old_id is not None and not np.array_equal(candidate[0],rows[0]):old_changes+=1
        assert (cache.counts<cfg['k']).all()
        steps.append(dict(index=index,entity=entity,mutable_ids=ids,weights=weights,radii=radii,receipt=receipt,
                          full_scope_max_rank=int(cache.counts.max())+1,cache_version=cache.version,
                          committed_state_sha256=__import__('hashlib').sha256(cache.E.cpu().numpy().tobytes()).hexdigest()))
        ids_records.append(ids);before.append(rows);after.append(candidate)
        if preflight or (index+1)%cfg['full_recount_every_updates']==0:cache.full_recount()
        if (index+1)%cfg['progress_every_updates']==0:print('SHARED_GRAPH_PROGRESS',job['id'],index+1,len(order),accepted,fallback,old_changes,round(time.monotonic()-started,2),flush=True)
    terminal=cache.full_recount();E=cache.E
    nonselected=np.setdiff1d(np.arange(7505),selected);assert torch.equal(E[torch.as_tensor(nonselected,device=device)],initial[torch.as_tensor(nonselected,device=device)])
    validation=None
    if not preflight:
        valid=[data.triples(s,'valid') for s in (0,1)];vg=torch.Generator().manual_seed(cfg['validation_seed'])
        valid=[q[torch.randperm(len(q),generator=vg)[:cfg['validation_queries_per_split']]] for q in valid]
        source_ranks=[ranks(target,R,q,'DistMult') for q in valid];actual_ranks=[ranks(E,R,q,'DistMult') for q in valid]
        old_before=ranks(initial[:7505],R,valid[0],'DistMult');protected=old_before<=10
        cohorts={'query_head_new_gold_old':(valid[1][:,0]>=7505)&(valid[1][:,2]<7505),'query_head_old_gold_new':(valid[1][:,0]<7505)&(valid[1][:,2]>=7505),'both_new':(valid[1][:,0]>=7505)&(valid[1][:,2]>=7505)}
        validation=dict(source=[metrics(r) for r in source_ranks],after=[metrics(r) for r in actual_ranks],
                        source_old_top10_loss=float((source_ranks[0][protected]>10).mean()),after_old_top10_loss=float((actual_ranks[0][protected]>10).mean()),
                        directional={k:dict(queries=int(m.sum()),source=metrics(source_ranks[1][m]),after=metrics(actual_ranks[1][m])) for k,m in cohorts.items()})
    torch.save(dict(E=E.cpu(),R=R.cpu(),kind=job['kind'],method=job['method'],seed=job['seed'],epoch=1),out/'best.pt')
    torch.save(dict(E=initial.cpu(),R=R.cpu(),oldE=initial[:7505].cpu()),out/'initial.pt')
    # Variable one/two-row patches are padded with entity ID-1 and zero values.
    replay_ids=np.full((len(order),2),-1,dtype=np.int64);replay_before=np.zeros((len(order),2,200));replay_after=np.zeros_like(replay_before)
    for i,ids in enumerate(ids_records):replay_ids[i,:len(ids)]=ids;replay_before[i,:len(ids)]=before[i];replay_after[i,:len(ids)]=after[i]
    np.savez_compressed(out/'shared-replay.npz',entity_order=order,mutable_ids=replay_ids,before=replay_before,after=replay_after,terminal_counts=cache.counts.cpu().numpy())
    (out/'steps.json').write_text(json.dumps(steps));(out/'selection.json').write_text(json.dumps(selection,indent=2));(out/'training.json').write_text(json.dumps([dict(epoch=1,validation=validation)]))
    bound.update(data.bound)
    for name,digest in bound.items():assert sha(Path(name))==digest
    result=dict(id=job['id'],method=job['method'],kind=job['kind'],seed=job['seed'],test_read=False,preflight=preflight,epochs_executed=1,
                elapsed_s=time.monotonic()-started,source_sgd_elapsed_s=source_receipt['elapsed_s'],processed_new_entities=len(order),accepted_updates=accepted,
                zero_fallbacks=fallback,steps_modifying_old_rows=old_changes,selected_old_rows=selected,selection=selection,validation=validation,
                guard_scope=job['protection_scope'],guard_arithmetic='Explicit float64 binary tree; common evaluation uses unchanged GEMM',
                terminal_complete_guard_recount=terminal,parameters=11258*200+474*200,trainable_parameters=0,modifiable_parameters=(3753+len(selected))*200,
                bound_input_sha256=bound,checkpoint_sha256=sha(out/'best.pt'),output_sha256={name:sha(out/name) for name in ['initial.pt','shared-replay.npz','steps.json','selection.json']},
                implementation_sha256=sha(Path(__file__)),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                limits=['Known mechanism combination, not confirmed novelty.','TRAIN compiled initially retained references only; no hard guarantee on all TEST.',
                        'All common evaluation queries retained; affected P exemption explicitly changes contract.',
                        'Working-set failures return unchanged; no global optimum guarantee.','Preflight performance is not a scientific utility outcome.'])
    (out/'training-receipt.json').write_text(json.dumps(result,indent=2));print('SHARED_GRAPH_TERMINAL',job['id'],len(order),accepted,fallback,old_changes,flush=True)
