"""TRAIN-only compilation of full/scoped shared-reference guards and cost audit."""
import json,time
from pathlib import Path
import numpy as np
import torch
from common_incremental import Data
from common_training import sha
from shared_rank_cache_v2 import fixed_scores


def source_initial(root,cfg,seed,device):
    source=root/'artifacts/incremental'/cfg['source_template'].format(seed=seed)
    adapted=root/'artifacts/incremental'/cfg['verified_proposal_template'].format(seed=seed)
    audit=json.loads((source/'audit.json').read_text());assert audit['status'].startswith('PASS') and not audit.get('failures')
    verified=json.loads((adapted/'audit.json').read_text());assert verified['status'].startswith('PASS') and not verified.get('failures')
    tr=json.loads((adapted/'training-receipt.json').read_text());initial=source/'initial.pt'
    assert tr['bound_input_sha256'][str(initial)]==sha(initial)
    model=torch.load(initial,map_location=device,weights_only=True)
    E=model['E'].double();R=model['R'].double()
    assert E.shape==(11258,200) and R.shape==(474,200) and torch.isfinite(E).all() and torch.isfinite(R).all()
    return E,R,{str(p):sha(p) for p in [initial,source/'audit.json',adapted/'audit.json',adapted/'training-receipt.json']}


def compact(keys,refs):
    used=np.unique(refs[:,0]);mapping=np.full(len(keys),-1,dtype=np.int64);mapping[used]=np.arange(len(used))
    compact_refs=refs.copy();compact_refs[:,0]=mapping[refs[:,0]]
    assert (compact_refs[:,0]>=0).all()
    return keys[used],np.unique(compact_refs,axis=0),used


@torch.no_grad()
def prepare(root,cfg,job):
    started=time.monotonic();torch.set_num_threads(2);device=torch.device('cuda:0')
    source_cfg=json.loads((root/cfg['task_config']).read_text());data=Data(root,source_cfg)
    new=data.triples(1,'train').numpy();affected=set(map(tuple,new[:,:2]))
    E,R,bound=source_initial(root,cfg,job['seed'],device)
    feature=root/source_cfg['feature_directory']/f"s{job['seed']}";receipt=json.loads((feature/'receipt.json').read_text())
    assert receipt['status']=='PASS_TRAIN_ONLY_GUARDS' and not receipt['test_read']
    assert sha(feature/'guards.npz')==receipt['output_sha256']
    for name,digest in receipt['bound_input_sha256'].items():assert sha(Path(name))==digest
    with np.load(feature/'guards.npz') as z:keys=z['pred_queries'].copy();true=z['true_queries'].copy()
    query_index={tuple(k):i for i,k in enumerate(keys)}
    true_q=np.asarray([query_index[tuple(t[:2])] for t in true]);affected_mask=np.asarray([tuple(k) in affected for k in keys])
    Q=E[torch.as_tensor(keys[:,0],device=device)]*R[torch.as_tensor(keys[:,1],device=device)]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
    S=torch.empty((len(keys),len(E)),dtype=torch.float64,device=device)
    for off in range(0,len(keys),cfg['score_batch']):
        assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Registered TRAIN preparation budget exceeded'
        S[off:off+cfg['score_batch']]=fixed_scores(Q[off:off+cfg['score_batch']],E)
        if off%cfg['progress_every_queries']==0:print('SHARED_SCOPE_SCORE',job['id'],off,len(keys),round(time.monotonic()-started,2),flush=True)
    torch.cuda.synchronize();score_seconds=time.monotonic()-begin
    true_ranks=[]
    for off in range(0,len(true),cfg['rank_batch']):
        q=torch.as_tensor(true_q[off:off+cfg['rank_batch']],device=device);g=torch.as_tensor(true[off:off+cfg['rank_batch'],2],device=device)
        part=S[q];cut=part[torch.arange(len(q),device=device),g];true_ranks.append((1+(part>cut[:,None]).sum(1)).cpu().numpy())
    true_ranks=np.concatenate(true_ranks);true_eligible=true_ranks<=cfg['k']
    members=[];prediction_ranks=[]
    for off in range(0,len(keys),cfg['rank_batch']):
        s=S[off:off+cfg['rank_batch']]
        chosen=torch.argsort(s[:,:7505],dim=1,descending=True,stable=True)[:,:cfg['k']]
        cut=s.gather(1,chosen).min(1).values
        members.append(chosen.cpu().numpy());prediction_ranks.append((1+(s>cut[:,None]).sum(1)).cpu().numpy())
    members=np.concatenate(members);prediction_ranks=np.concatenate(prediction_ranks);pred_eligible=prediction_ranks<=cfg['k']
    trefs=np.column_stack([true_q[true_eligible],true[true_eligible,2]])
    pref_all=np.column_stack([np.repeat(np.flatnonzero(pred_eligible),cfg['k']),members[pred_eligible].ravel()])
    pref_scoped=np.column_stack([np.repeat(np.flatnonzero(pred_eligible&~affected_mask),cfg['k']),members[pred_eligible&~affected_mask].ravel()])
    full_keys,full_refs,full_ids=compact(keys,np.r_[trefs,pref_all])
    scope_keys,scope_refs,scope_ids=compact(keys,np.r_[trefs,pref_scoped])
    # Check the exact grouped minimum rule on all compiled reference members.
    def grouped(used,refs):
        qt=torch.as_tensor(used[refs[:,0]],device=device);gt=torch.as_tensor(refs[:,1],device=device)
        cut=torch.full((len(keys),),float('inf'),dtype=torch.float64,device=device)
        cut.scatter_reduce_(0,qt,S[qt,gt],reduce='amin',include_self=True)
        counts=[]
        for off in range(0,len(used),cfg['rank_batch']):
            q=torch.as_tensor(used[off:off+cfg['rank_batch']],device=device);counts.append((S[q]>cut[q,None]).sum(1).cpu().numpy())
        counts=np.concatenate(counts);assert (counts<cfg['k']).all()
        return counts
    full_counts=grouped(full_ids,full_refs);scope_counts=grouped(scope_ids,scope_refs)
    ids=np.sort(np.random.default_rng(cfg['fixture_seed']+job['seed']).choice(len(keys),cfg['fixture_queries'],replace=False))
    fixtures=S[torch.as_tensor(ids,device=device)].cpu().numpy()
    out=root/cfg['output_directory']/job['id'];out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'scope.npz',source_keys=keys,source_true=true,source_true_query_ids=true_q,source_true_initial_ranks=true_ranks,
                        pred_old_members=members,pred_initial_ranks=prediction_ranks,affected_mask=affected_mask,
                        full_keys=full_keys,full_refs=full_refs,full_source_query_ids=full_ids,full_counts=full_counts,
                        scoped_keys=scope_keys,scoped_refs=scope_refs,scoped_source_query_ids=scope_ids,scoped_counts=scope_counts)
    np.savez_compressed(out/'fixtures.npz',query_ids=ids,scores=fixtures)
    bound.update(data.bound);bound.update({str(p):sha(p) for p in [feature/'guards.npz',feature/'receipt.json',root/cfg['task_config']]})
    for name,digest in bound.items():assert sha(Path(name))==digest
    result=dict(id=job['id'],seed=job['seed'],scope='TRAIN-only shared-reference guard compilation and full candidate scaling; no new method utility',
                test_read=False,valid_read=False,new_train_read=True,new_entities=3753,candidates=len(E),dimension=E.shape[1],
                source_true_rows=len(true),true_initially_retained=int(true_eligible.sum()),true_initially_ineligible=int((~true_eligible).sum()),
                source_prediction_queries=len(keys),prediction_initially_retained_queries=int(pred_eligible.sum()),
                prediction_initially_ineligible_queries=int((~pred_eligible).sum()),affected_old_train_query_keys=int(affected_mask.sum()),
                full_reference_members=len(full_refs),full_grouped_queries=len(full_keys),scoped_reference_members=len(scope_refs),scoped_grouped_queries=len(scope_keys),
                score_matrix_bytes=S.numel()*S.element_size(),full_candidate_scoring_s=score_seconds,elapsed_s=time.monotonic()-started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),fixture_queries=len(ids),torch_version=torch.__version__,gpu_name=torch.cuda.get_device_name(),source_information='Audited common initial model; existing development foundation/source checkpoints',
                input_sha256=bound,output_sha256={name:sha(out/name) for name in ['scope.npz','fixtures.npz']},
                implementation_sha256=sha(Path(__file__)),
                limits=['Preparation/scaling only, no graph optimization or model utility result.',
                        'All initially retained source-selected TRUE anchors; original TRAIN per-relation cap remains.',
                        'Fixed binary-tree FP64 guard scoring differs from prior GEMM; existing common evaluation remains unchanged.',
                        'Scoped P explicitly exempts newTRAIN query keys; all TRUE anchors and final common evaluation queries retained.'])
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print('SHARED_SCOPE_READY',json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','output_sha256','limits']}),flush=True)


def audit(root,cfg,job):
    started=time.monotonic();torch.set_num_threads(2);out=root/cfg['output_directory']/job['id'];receipt=json.loads((out/'receipt.json').read_text())
    assert not receipt['test_read'] and not receipt['valid_read']
    for name,digest in receipt['input_sha256'].items():assert sha(Path(name))==digest
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest
    E,R,_=source_initial(root,cfg,job['seed'],'cpu');E=E.numpy();R=R.numpy()
    source_cfg=json.loads((root/cfg['task_config']).read_text());data=Data(root,source_cfg);new=data.triples(1,'train').numpy();affected=set(map(tuple,new[:,:2]))
    with np.load(out/'scope.npz') as z, np.load(out/'fixtures.npz') as f:
        keys=z['source_keys'];true=z['source_true'];true_q=z['source_true_query_ids'];tranks=z['source_true_initial_ranks'];members=z['pred_old_members'];pranks=z['pred_initial_ranks'];mask=z['affected_mask']
        feature=root/source_cfg['feature_directory']/f"s{job['seed']}"
        with np.load(feature/'guards.npz') as source:assert np.array_equal(keys,source['pred_queries']) and np.array_equal(true,source['true_queries'])
        assert np.array_equal(keys[true_q],true[:,:2])
        assert np.array_equal(mask,[tuple(k) in affected for k in keys])
        trefs=np.column_stack([true_q[tranks<=cfg['k']],true[tranks<=cfg['k'],2]])
        for mode in ['full','scoped']:
            included=pranks<=cfg['k']
            if mode=='scoped':included=included&~mask
            predrefs=np.column_stack([np.repeat(np.flatnonzero(included),cfg['k']),members[included].ravel()])
            wanted=np.unique(np.r_[trefs,predrefs],axis=0);used=np.unique(wanted[:,0]);mapping={q:i for i,q in enumerate(used)}
            wanted[:,0]=np.asarray([mapping[q] for q in wanted[:,0]])
            assert np.array_equal(z[mode+'_keys'],keys[used]) and np.array_equal(z[mode+'_refs'],wanted)
            assert len(z[mode+'_counts'])==len(used) and (z[mode+'_counts']<cfg['k']).all()
        max_error=0.;padded=1<<(E.shape[1]-1).bit_length();checked=0
        for qi,saved in zip(f['query_ids'],f['scores']):
            h,r=keys[qi];direction=E[h]*R[r]
            tree=np.zeros((len(E),padded));tree[:,:E.shape[1]]=E*direction
            while tree.shape[1]>1:tree=tree[:,0::2]+tree[:,1::2]
            scores=tree[:,0];error=float(np.max(np.abs(scores-saved)));max_error=max(max_error,error)
            assert np.array_equal(scores,saved),'CPU binary-tree full row differs from CUDA fixture'
            chosen=np.asarray(sorted(range(7505),key=lambda g:(-scores[g],g))[:cfg['k']]);assert np.array_equal(chosen,members[qi])
            assert int(1+(scores>scores[chosen].min()).sum())==pranks[qi]
            selected=np.flatnonzero(true_q==qi)
            for index in selected:assert int(1+(scores>scores[true[index,2]]).sum())==tranks[index]
            checked+=len(scores)
    result=dict(status='PASS_TRAIN_SCOPE_IDENTITY_ALL_MEMBERS_AND_FULL_POOL_CPU_FIXTURES',failures=[],
                source_true_rows_checked=len(true),prediction_member_identities_checked=int(members.size),
                full_candidate_cpu_scores_checked=checked,max_cpu_gpu_score_error=max_error,
                complete_reference_scope_reconstructed=True,affected_keys_train_only=True,test_read=False,valid_read=False,
                elapsed_s=time.monotonic()-started,receipt_sha256=sha(out/'receipt.json'),implementation_sha256=sha(Path(__file__)),
                limits=['Full structural scope checked; independent score replay is fixed128 full query rows, not all68334 rows.',
                        'No trained redistribution or new method utility is established.'])
    (out/'audit.json').write_text(json.dumps(result,indent=2));print('SHARED_SCOPE_AUDIT',job['id'],json.dumps(result),flush=True)
