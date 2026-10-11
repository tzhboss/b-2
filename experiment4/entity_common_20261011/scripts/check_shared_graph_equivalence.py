"""Replay audited real TRAIN fixtures against interchangeable shared solvers."""
import argparse,importlib,json,sys,time
from pathlib import Path
from hashlib import sha256
import numpy as np
import torch
import yaml


def digest(path):return sha256(path.read_bytes()).hexdigest()


def inputs(root,cfg,job):
    canonical=yaml.safe_load((root/cfg['source_canonical']).read_text())
    assert canonical['status']=='completed_audited_artifact_checks_only'
    assert canonical['jobs'][job['source_job']]['status']=='completed_audited'
    assert digest(root/canonical['config'])==cfg['source_config_sha256']
    source_cfg=json.loads((root/canonical['config']).read_text())
    source_job=next(j for j in source_cfg['jobs'] if j['id']==job['source_job'])
    folder=root/cfg['source_directory']/job['source_job']
    receipt=json.loads((folder/'training-receipt.json').read_text());audit=json.loads((folder/'audit.json').read_text())
    assert receipt['preflight'] and not receipt['test_read'] and receipt['validation'] is None
    assert receipt['processed_new_entities']==cfg['expected_steps']
    assert audit['status']=='PASS_PREFLIGHT_PATCH_REPLAY_AND_ALL_QUERY_CPU_TERMINAL_RANKS' and not audit['failures']
    assert audit['receipt_sha256']==digest(folder/'training-receipt.json') and audit['terminal_cpu_counts_exact']
    assert digest(folder/'best.pt')==receipt['checkpoint_sha256']
    for name,value in receipt['bound_input_sha256'].items():assert digest(Path(name))==value
    for name,value in receipt['output_sha256'].items():assert digest(folder/name)==value
    return source_cfg,source_job,folder,receipt,audit


@torch.no_grad()
def prepare(root,cfg,job):
    from shared_rank_graph import load_scope
    from shared_rank_cache_v2 import SharedRankCache
    started=time.monotonic();source_cfg,source_job,folder,source_receipt,source_audit=inputs(root,cfg,job)
    dest=root/cfg['output_directory']/job['id'];dest.mkdir(parents=True,exist_ok=False)
    initial=torch.load(folder/'initial.pt',map_location='cuda:0',weights_only=True)
    keys,refs,bound=load_scope(root,source_cfg,source_job['seed'],source_job['protection_scope'])
    cache=SharedRankCache(initial['E'],initial['R'],keys,refs,source_cfg['k'],source_cfg['score_batch'])
    target_path=root/'artifacts/incremental'/source_cfg['source_template'].format(seed=source_job['seed'])/'best.pt'
    target=torch.load(target_path,map_location='cpu',weights_only=True)['E'].numpy().astype(np.float64)
    steps=json.loads((folder/'steps.json').read_text());records=[];times={name:[] for name in cfg['solver_modules']}
    solvers={name:importlib.import_module(name).solve_shared_update for name in cfg['solver_modules']}
    with np.load(folder/'shared-replay.npz') as z:
        assert len(steps)==len(z['entity_order'])==cfg['expected_steps']
        for i,step in enumerate(steps):
            assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Registered replay budget exceeded'
            ids=z['mutable_ids'][i];ids=ids[ids>=0].tolist();entity=int(z['entity_order'][i])
            assert step['entity']==entity and step['mutable_ids']==ids
            assert np.array_equal(cache.E[ids].cpu().numpy(),z['before'][i,:len(ids)])
            centers=initial['E'][ids].cpu().numpy().copy();desired=cache.E[ids].cpu().numpy().copy();desired[-1]=target[entity]
            if len(ids)==2:desired[0]=centers[0]
            paired=[]
            # Alternate order to avoid assigning all warm-cache advantage to V3.
            names=cfg['solver_modules'][::1 if i%2==0 else -1]
            for name in names:
                torch.cuda.synchronize();t=time.monotonic()
                patch,receipt=solvers[name](cache,ids,desired,step['weights'],centers,step['radii'],ids[:-1],source_cfg['solver'])
                torch.cuda.synchronize();times[name].append(time.monotonic()-t);paired.append((patch,receipt))
            pa,ra=paired[0];pb,rb=paired[1]
            clean=lambda r:{k:v for k,v in r.items() if k!='elapsed_s'}
            assert clean(ra)==clean(rb)==clean(step['receipt'])
            for name in ['ids','rows','columns','head_queries','head_directions','head_scores','thresholds','counts']:
                assert torch.equal(getattr(pa,name),getattr(pb,name)),(i,name)
            assert np.array_equal(pa.rows.cpu().numpy(),z['after'][i,:len(ids)])
            if ra['accepted']:assert cache.commit(pa)
            else:assert np.array_equal(pa.rows.cpu().numpy(),cache.E[ids].cpu().numpy())
            state_hash=sha256(cache.E.cpu().numpy().tobytes()).hexdigest();assert state_hash==step['committed_state_sha256']
            records.append(dict(index=i,entity=entity,mutable_ids=ids,receipt=clean(ra),state_sha256=state_hash,patch_tensors_bitwise_equal=True))
            print('REAL_GRAPH_EQUIVALENCE_STEP',job['id'],i+1,len(steps),flush=True)
        terminal=cache.full_recount();assert np.array_equal(cache.counts.cpu().numpy(),z['terminal_counts'])
    final=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert torch.equal(cache.E.cpu(),final['E']) and torch.equal(cache.R.cpu(),final['R'])
    np.savez_compressed(dest/'terminal.npz',E=cache.E.cpu().numpy(),R=cache.R.cpu().numpy(),counts=cache.counts.cpu().numpy())
    (dest/'steps.json').write_text(json.dumps(records))
    bound.update({str(folder/name):digest(folder/name) for name in ['initial.pt','best.pt','training-receipt.json','audit.json','steps.json','shared-replay.npz']})
    bound[str(target_path)]=digest(target_path)
    result=dict(status='PASS_REAL_TRAIN_FIXTURE_SOLVER_BITWISE_EQUIVALENCE',source_job=job['source_job'],steps=len(records),
        patch_tensor_fields_checked=9*len(records),source_cpu_terminal_rank_audit=source_audit['status'],
        terminal_guard_recount=terminal,paired_solver_wall_seconds=times,bound_input_sha256=bound,
        output_sha256={name:digest(dest/name) for name in ['terminal.npz','steps.json']},
        test_read=False,valid_read=False,utility_measured=False,full3753_equivalence=False,elapsed_s=time.monotonic()-started,
        limits=['Audited eight-step real TRAIN fixture only; no complete3753 utility or end-to-end speedup.',
                'Exact source terminal weights reuse the existing independent CPU terminal rank audit; no new full CPU scoring.',
                'Paired solver timings alternate call order, one device and seed; descriptive only.'])
    for name,value in bound.items():assert digest(Path(name))==value
    (dest/'receipt.json').write_text(json.dumps(result,indent=2))


def audit(root,cfg,job):
    source_cfg,source_job,folder,source_receipt,source_audit=inputs(root,cfg,job)
    dest=root/cfg['output_directory']/job['id'];receipt=json.loads((dest/'receipt.json').read_text())
    assert receipt['status']=='PASS_REAL_TRAIN_FIXTURE_SOLVER_BITWISE_EQUIVALENCE'
    for name,value in receipt['bound_input_sha256'].items():assert digest(Path(name))==value
    for name,value in receipt['output_sha256'].items():assert digest(dest/name)==value
    steps=json.loads((dest/'steps.json').read_text());source_steps=json.loads((folder/'steps.json').read_text())
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True);E=initial['E'].numpy().copy();R=initial['R'].numpy()
    with np.load(folder/'shared-replay.npz') as z:
        for i,step in enumerate(steps):
            old=source_steps[i];ids=z['mutable_ids'][i];ids=ids[ids>=0]
            assert step['mutable_ids']==ids.tolist() and step['entity']==int(z['entity_order'][i])
            assert step['receipt']=={k:v for k,v in old['receipt'].items() if k!='elapsed_s'}
            assert np.array_equal(E[ids],z['before'][i,:len(ids)])
            E[ids]=z['after'][i,:len(ids)]
            assert np.all(np.linalg.norm(E[ids]-initial['E'][ids].numpy(),axis=1)<=np.asarray(old['radii'])+source_cfg['solver']['movement_tolerance'])
            assert sha256(E.tobytes()).hexdigest()==step['state_sha256']==old['committed_state_sha256']
        with np.load(dest/'terminal.npz') as actual:
            assert np.array_equal(actual['E'],E) and np.array_equal(actual['R'],R) and np.array_equal(actual['counts'],z['terminal_counts'])
    assert len(steps)==cfg['expected_steps']==source_audit['patches_replayed']
    result=dict(status='PASS_REAL_FIXTURE_TRAJECTORY_IDENTITIES_AND_INHERITED_CPU_SCOPE',failures=[],steps_replayed=len(steps),
        source_cpu_scope_queries=source_audit['terminal_query_groups_replayed'],source_cpu_scope_reference_members=source_audit['reference_members_retained'],
        source_cpu_audit_sha256=digest(folder/'audit.json'),receipt_sha256=digest(dest/'receipt.json'),
        test_read=False,valid_read=False,utility_measured=False,limits=receipt['limits'])
    (dest/'audit.json').write_text(json.dumps(result,indent=2));print('REAL_GRAPH_EQUIVALENCE_AUDIT_PASS',job['id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['prepare','audit'],required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));torch.set_num_threads(2);cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='prepare':prepare(root,cfg,job)
    else:audit(root,cfg,job)
