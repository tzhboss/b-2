"""Recompute full shared updates and bind every step to audited source weights."""
import argparse,importlib,json,sys
from pathlib import Path
from hashlib import sha256
import numpy as np
import torch
import yaml


def digest(path):return sha256(path.read_bytes()).hexdigest()


def source(root,cfg,job):
    canonical=yaml.safe_load((root/cfg['source_job_dependency']).read_text())
    assert canonical['jobs'][job['id']]['status']=='completed_audited'
    assert canonical['config']==cfg['source_config'] and digest(root/canonical['config'])==cfg['source_config_sha256']
    original=json.loads((root/canonical['config']).read_text());original_job=next(j for j in original['jobs'] if j['id']==job['id']);assert job==original_job
    folder=root/'artifacts/incremental'/job['id'];tr=json.loads((folder/'training-receipt.json').read_text());a=json.loads((folder/'audit.json').read_text())
    assert not tr['preflight'] and not tr['test_read'] and tr['processed_new_entities']==cfg['expected_steps']==3753
    assert a['status']=='PASS_COMMON_INFERENCE_ALL3753_PATCHES_AND_FULL_TERMINAL_SCOPE' and not a['failures']
    assert a['solver_audit']['status']=='PASS_ALL3753_PATCHES_AND_ALL_QUERY_CPU_TERMINAL_RANKS'
    assert a['solver_audit']['terminal_cpu_counts_exact'] and a['solver_audit']['patches_replayed']==3753
    assert digest(folder/'best.pt')==tr['checkpoint_sha256']
    for name,value in tr['bound_input_sha256'].items():assert digest(Path(name))==value
    for name,value in tr['output_sha256'].items():assert digest(folder/name)==value
    return original,folder,tr,a


def clean_step(row):
    value=dict(row);value['receipt']={k:v for k,v in row['receipt'].items() if k!='elapsed_s'};return value


def prepare(root,cfg,job):
    import shared_rank_graph as graph
    original,folder,source_receipt,source_audit=source(root,cfg,job)
    bound={str(folder/name):digest(folder/name) for name in ['best.pt','initial.pt','shared-replay.npz','steps.json','selection.json','training-receipt.json','audit.json']}
    dest=root/cfg['output_directory']/job['id'];dest.mkdir(parents=True,exist_ok=False)
    runtime_job=dict(job,id=cfg['runtime_job_prefix']+job['id']);out=root/'artifacts/incremental'/runtime_job['id'];assert not out.exists()
    saved=np.load(folder/'shared-replay.npz');steps=json.loads((folder/'steps.json').read_text());assert len(steps)==3753
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
    algorithm=importlib.import_module(cfg['solver_module']).solve_shared_update
    n=0;previous=graph.solve_shared_update;solver_seconds=[]
    def qualified(cache,ids,target,weights,centers,radii,preferred_old,options):
        nonlocal n
        i=n;expected=steps[i];selected=saved['mutable_ids'][i];selected=selected[selected>=0]
        assert ids==selected.tolist()==expected['mutable_ids'] and preferred_old==ids[:-1]
        assert weights==expected['weights'] and radii==expected['radii'] and options==original['solver']
        assert np.array_equal(cache.E[ids].cpu().numpy(),saved['before'][i,:len(ids)])
        assert np.array_equal(centers,initial['E'][ids].numpy())
        if i==0:assert torch.equal(cache.E.cpu(),initial['E']) and torch.equal(cache.R.cpu(),initial['R'])
        patch,receipt=algorithm(cache,ids,target,weights,centers,radii,preferred_old,options)
        assert {k:v for k,v in receipt.items() if k!='elapsed_s'}=={k:v for k,v in expected['receipt'].items() if k!='elapsed_s'},i
        assert np.array_equal(patch.rows.cpu().numpy(),saved['after'][i,:len(ids)]),i
        assert int(patch.counts.max())+1==expected['full_scope_max_rank']
        assert cache.version+int(receipt['accepted'])==expected['cache_version']
        candidate=cache.E.cpu().numpy().copy();candidate[ids]=patch.rows.cpu().numpy()
        assert sha256(candidate.tobytes()).hexdigest()==expected['committed_state_sha256'],i
        solver_seconds.append(receipt['elapsed_s']);n+=1;return patch,receipt
    graph.solve_shared_update=qualified
    try:graph.train_rank(root,original,runtime_job,preflight=False)
    finally:graph.solve_shared_update=previous;saved.close()
    assert n==3753
    tr=json.loads((out/'training-receipt.json').read_text())
    expected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);actual=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    assert torch.equal(actual['E'],expected['E']) and torch.equal(actual['R'],expected['R'])
    assert tr['validation']==source_receipt['validation']
    scope=json.loads((root/original['scope_directory']/f"shared-rank-scope-s{job['seed']}"/'receipt.json').read_text())
    tr.update(solver_module=cfg['solver_module'],qualification_worker_sha256=digest(Path(__file__)),
              scope_preparation_elapsed_s=scope['elapsed_s'],source_inclusive_incremental_elapsed_s=tr['elapsed_s']+tr['source_sgd_elapsed_s']+scope['elapsed_s'])
    (out/'training-receipt.json').write_text(json.dumps(tr,indent=2))
    for name,value in bound.items():assert digest(Path(name))==value
    receipt=dict(status='PASS_ALL3753_CACHED_SOLVER_SOURCE_TRAJECTORY',id=job['id'],runtime_id=runtime_job['id'],method=job['method'],seed=job['seed'],
        steps_matched=n,terminal_weights_bitwise_equal=True,fixed_validation_identical=True,source_original_cpu_scope=a_scope(source_audit),
        source_postprocessing_elapsed_s=source_receipt['elapsed_s'],cached_postprocessing_elapsed_s=tr['elapsed_s'],
        source_inclusive_incremental_elapsed_s=source_receipt['source_inclusive_incremental_elapsed_s'],cached_inclusive_incremental_elapsed_s=tr['source_inclusive_incremental_elapsed_s'],
        cached_solver_elapsed_sum_s=sum(solver_seconds),bound_source_sha256=bound,
        bound_runtime_sha256={str(out/name):digest(out/name) for name in ['best.pt','initial.pt','shared-replay.npz','steps.json','selection.json','training-receipt.json']},
        test_read=False,valid_read=True,validation_selection_changed=False,utility_search=False,
        limits=['Complete same-math postprocessing recomputation, not new source SGD training or a new method slot.',
                'Exact source weights inherit030 TEST/CPU rank evidence; no new independent full TEST scoring.',
                'One repetition per seed/device; compare declared whole postprocessing costs, not index-only timing.',
                'No utility gates, coefficients, protected queries or parameter budgets changed.'])
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print('FULL_CACHED_SOLVER_TRAJECTORY_MATCH',job['id'],n,flush=True)


def a_scope(audit):return {k:audit['solver_audit'][k] for k in ['status','terminal_query_groups_replayed','reference_members_retained','terminal_cpu_counts_exact']}


def audit(root,cfg,job):
    original,folder,tr,source_audit=source(root,cfg,job);dest=root/cfg['output_directory']/job['id'];r=json.loads((dest/'receipt.json').read_text())
    for name,value in r['bound_source_sha256'].items():assert digest(Path(name))==value
    for name,value in r['bound_runtime_sha256'].items():assert digest(Path(name))==value
    out=root/'artifacts/incremental'/r['runtime_id'];new_tr=json.loads((out/'training-receipt.json').read_text())
    with np.load(folder/'shared-replay.npz') as old,np.load(out/'shared-replay.npz') as new:
        assert set(old.files)==set(new.files)
        for key in old.files:assert np.array_equal(old[key],new[key]),key
    old_steps=json.loads((folder/'steps.json').read_text());new_steps=json.loads((out/'steps.json').read_text())
    assert len(old_steps)==len(new_steps)==r['steps_matched']==3753
    assert [clean_step(s) for s in old_steps]==[clean_step(s) for s in new_steps]
    for name in ['initial.pt','best.pt']:
        old=torch.load(folder/name,map_location='cpu',weights_only=True);new=torch.load(out/name,map_location='cpu',weights_only=True)
        assert old.keys()==new.keys()
        for key in old:assert torch.equal(old[key],new[key]) if torch.is_tensor(old[key]) else old[key]==new[key]
    assert json.loads((out/'selection.json').read_text())==json.loads((folder/'selection.json').read_text())
    for key in ['seed','method','kind','test_read','validation','selected_old_rows','selection','processed_new_entities','accepted_updates','zero_fallbacks','steps_modifying_old_rows','guard_scope','guard_arithmetic','terminal_complete_guard_recount','parameters','modifiable_parameters','source_sgd_elapsed_s','scope_preparation_elapsed_s']:
        assert new_tr[key]==tr[key],key
    assert not new_tr['test_read'] and new_tr['solver_module']==cfg['solver_module']
    result=dict(status='PASS_ALL3753_INDEPENDENT_TRACE_IDENTITY_AND_INHERITED_FULL_CPU_SCOPE',failures=[],steps_replayed=3753,
        exact_all_state_hashes=True,exact_all_replay_arrays=True,terminal_weights_bitwise_equal=True,
        source_cpu_scope=a_scope(source_audit),source_audit_sha256=digest(folder/'audit.json'),receipt_sha256=digest(dest/'receipt.json'),
        test_read=False,valid_read=True,utility_search=False,limits=r['limits'])
    (dest/'audit.json').write_text(json.dumps(result,indent=2));print('FULL_CACHED_SOLVER_INDEPENDENT_AUDIT_PASS',job['id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['prepare','audit'],required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));torch.set_num_threads(2);cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='prepare':prepare(root,cfg,job)
    else:audit(root,cfg,job)
