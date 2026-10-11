"""Independent complete shared-graph patch and terminal-rank audit."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch


def independent_budget(data,keys,refs,E,R,target,options,k):
    """Rebuild TRAIN selection with NumPy scores, without producer selectors."""
    from collections import defaultdict
    old=data.triples(0,'train').numpy();new=data.triples(1,'train').numpy()
    degree=np.zeros(7505,dtype=np.int64);positive=defaultdict(set)
    for h,r,t in old:degree[h]+=1;positive[(int(h),int(r))].add(int(t))
    query_index={tuple(map(int,key)):i for i,key in enumerate(keys)}
    members=defaultdict(list)
    for q,g in refs:members[int(q)].append(int(g))
    entity_queries=defaultdict(set)
    for h,r,t in new:
        if h<7505 and t>=7505 and (int(h),int(r)) in query_index:entity_queries[int(t)].add((int(h),int(r)))
    relevant=sorted(set(q for qs in entity_queries.values() for q in qs));score_rows={}
    padded=1<<(E.shape[1]-1).bit_length()
    for h,r in relevant:
        qi=query_index[(h,r)];tree=np.zeros((len(E),padded));tree[:,:E.shape[1]]=E*(E[h]*R[r])
        while tree.shape[1]>1:tree=tree[:,0::2]+tree[:,1::2]
        scores=tree[:,0];boundary=min(scores[g] for g in members[qi])
        if int((scores>boundary).sum())==k-1:score_rows[(h,r)]=(scores[:7505].copy(),boundary)
    voters=defaultdict(set)
    for entity,queries in entity_queries.items():
        for h,r in sorted(queries):
            if (h,r) not in score_rows:continue
            scores,boundary=score_rows[(h,r)]
            if float(np.dot(E[h]*R[r],target[entity]))<=boundary:continue
            candidates=[int(c) for c in np.flatnonzero(scores>boundary) if int(c) not in positive[(h,r)] and int(c)!=h]
            if candidates:voters[min(candidates,key=lambda c:(scores[c]-boundary,degree[c],c))].add(entity)
    selected=sorted(voters,key=lambda c:(-len(voters[c]),degree[c],c))[:options['max_old_rows']]
    choices=defaultdict(set)
    for candidate in selected:
        for entity in voters[candidate]:choices[entity].add(candidate)
    return selected,{e:min(cs,key=lambda c:(degree[c],c)) for e,cs in choices.items()},degree


def audit(root,cfg,job):
    from common_training import sha
    from shared_rank_graph import load_scope
    from common_incremental import Data
    torch.set_num_threads(2);started=time.monotonic();out=root/'artifacts/incremental'/job['id']
    receipt=json.loads((out/'training-receipt.json').read_text());assert not receipt['preflight'] and not receipt['test_read'] and receipt['validation'] is not None
    for name,digest in receipt['bound_input_sha256'].items():assert sha(Path(name))==digest
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest
    assert sha(out/'best.pt')==receipt['checkpoint_sha256']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True);E=initial['E'].numpy().copy();R=initial['R'].numpy();centers=E.copy()
    keys,refs,_=load_scope(root,cfg,job['seed'],job['protection_scope']);steps=json.loads((out/'steps.json').read_text())
    source=root/'artifacts/incremental'/cfg['source_template'].format(seed=job['seed'])
    target=torch.load(source/'best.pt',map_location='cpu',weights_only=True)['E'].numpy().astype(np.float64)
    if job['joint_old']:
        selected_old,per_entity,degree=independent_budget(Data(root,cfg),keys,refs,E,R,target,cfg['parameter_budget'],cfg['k'])
    else:selected_old,per_entity,degree=[],{},np.zeros(7505)
    assert receipt['selected_old_rows']==selected_old
    radius=float(np.linalg.norm(centers,axis=1).max());median_degree=float(np.median(degree[degree>0])) if selected_old else 1.
    with np.load(out/'shared-replay.npz') as z:
        order=z['entity_order'];ids=z['mutable_ids'];before=z['before'];after=z['after'];assert len(order)==len(steps)==3753
        expected=(torch.randperm(3753,generator=torch.Generator().manual_seed(job['seed']+271))+7505).numpy()
        assert np.array_equal(order,expected) and np.array_equal(np.sort(order),np.arange(7505,11258))
        old_ids=set(receipt['selected_old_rows']);checked=0
        for i,row in enumerate(steps):
            selected=ids[i][ids[i]>=0];assert selected.tolist()==row['mutable_ids'] and selected[-1]==order[i]==row['entity']
            old_id=per_entity.get(int(order[i]));expected_ids=[int(order[i])] if old_id is None else [old_id,int(order[i])]
            assert selected.tolist()==expected_ids
            expected_weights=[1.] if old_id is None else [cfg['parameter_budget']['old_penalty']*(1+degree[old_id]/median_degree),1.]
            expected_radii=[2*radius] if old_id is None else [cfg['parameter_budget']['old_radius_fraction']*float(np.linalg.norm(centers[old_id])),2*radius]
            assert row['weights']==expected_weights and np.allclose(row['radii'],expected_radii,rtol=1e-14,atol=1e-14)
            assert np.array_equal(E[selected],before[i,:len(selected)])
            if len(selected)==2:assert selected[0] in old_ids
            if not row['receipt']['accepted']:assert np.array_equal(before[i,:len(selected)],after[i,:len(selected)])
            E[selected]=after[i,:len(selected)]
            move=np.linalg.norm(E[selected]-centers[selected],axis=1);assert np.all(move<=np.asarray(expected_radii)+cfg['solver']['movement_tolerance'])
            assert hashlib.sha256(E.tobytes()).hexdigest()==row['committed_state_sha256']
            assert row['full_scope_max_rank']<=cfg['k'];checked+=len(selected)
        final=torch.load(out/'best.pt',map_location='cpu',weights_only=True);assert np.array_equal(E,final['E'].numpy()) and np.array_equal(R,final['R'].numpy())
        unused=np.setdiff1d(np.arange(7505),list(old_ids));assert np.array_equal(E[unused],centers[unused])
        # Independent CPU tree across every reference query at the terminal model.
        by_query={}
        for q,g in refs:by_query.setdefault(int(q),[]).append(int(g))
        counts=[];padded=1<<(E.shape[1]-1).bit_length()
        for qi,(h,r) in enumerate(keys):
            tree=np.zeros((len(E),padded));tree[:,:E.shape[1]]=E*(E[h]*R[r])
            while tree.shape[1]>1:tree=tree[:,0::2]+tree[:,1::2]
            scores=tree[:,0];cut=min(scores[g] for g in by_query[qi]);count=int((scores>cut).sum());assert count<cfg['k'],(qi,count)
            counts.append(count)
            if (qi+1)%8192==0:print('SHARED_PREFLIGHT_CPU_SCOPE',job['id'],qi+1,len(keys),round(time.monotonic()-started,2),flush=True)
        assert np.array_equal(counts,z['terminal_counts'])
    result=dict(status='PASS_ALL3753_PATCHES_AND_ALL_QUERY_CPU_TERMINAL_RANKS',failures=[],preflight=False,
                patches_replayed=len(steps),modified_row_steps_replayed=checked,terminal_query_groups_replayed=len(keys),reference_members_retained=len(refs),
                terminal_cpu_counts_exact=True,absolute_budgets_checked=True,train_only_parameter_selection_rebuilt=True,weights_and_radii_independently_checked=True,unselected_old_rows_unchanged=True,test_read=False,valid_read=False,
                elapsed_s=time.monotonic()-started,receipt_sha256=sha(out/'training-receipt.json'),
                limits=['Complete patch identity/bounds replay and entire terminal scope; utility remains separate.',
                        'All-step GPU guards logged and terminal entire CPU scope replayed; intermediate CPU full scopes not repeated.'])
    (out/'solver-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True)
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job);sys.path.insert(0,str(root/'src'))
    audit(root,cfg,job)
