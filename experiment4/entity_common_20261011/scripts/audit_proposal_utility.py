"""Independent full-VALID recomputation and frozen development utility screen."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml


def numpy_ranks(E,R,q,batch=64):
    values=[]
    for off in range(0,len(q),batch):
        part=q[off:off+batch];s=(E[part[:,0]]*R[part[:,1]])@E.T;assert np.isfinite(s).all()
        gold=s[np.arange(len(part)),part[:,2]];values.append(1+(s>gold[:,None]).sum(1))
    return np.concatenate(values)


def audit(root,cfg,job):
    from common_incremental import Data,metrics
    from common_training import sha
    source_can=yaml.safe_load((root/'experiments'/cfg['source_experiment']/'experiment.yaml').read_text())
    assert sha(root/source_can['config'])==cfg['source_config_sha256']
    source_cfg=json.loads((root/source_can['config']).read_text());source_job=next(j for j in source_cfg['jobs'] if j['id']==job['id'])
    dest=root/cfg['reaudit_directory']/job['id'];dest.mkdir(parents=True,exist_ok=False)
    try:
        assert source_can['jobs'][job['id']]['status']=='completed_audited','Source method incomplete/failed; no utility assertion'
        out=root/'artifacts/incremental'/job['id'];prior=json.loads((out/'audit.json').read_text());assert not prior['failures'] and prior['solver_audit']['status']=='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS'
        tr=json.loads((out/'training-receipt.json').read_text());assert not tr['test_read'] and sha(out/'best.pt')==tr['checkpoint_sha256']
        for name,digest in tr['bound_input_sha256'].items():assert sha(name)==digest
        current=torch.load(out/'best.pt',map_location='cpu',weights_only=True);initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
        source_dir=root/'artifacts/incremental'/source_cfg['proposal_job_template'].format(seed=job['seed']);source=torch.load(source_dir/'best.pt',map_location='cpu',weights_only=True)
        E=current['E'].numpy();R=current['R'].numpy();Esource=source['E'].numpy().astype(np.float64);Rsource=source['R'].numpy().astype(np.float64)
        assert np.array_equal(R,Rsource) and np.array_equal(E[:7505],Esource[:7505]) and np.array_equal(initial['oldE'].numpy(),Esource[:7505])
        data=Data(root,source_cfg);vg=torch.Generator().manual_seed(source_cfg['validation_seed']);queries=[]
        for snapshot in (0,1):
            q=data.triples(snapshot,'valid').numpy();ids=torch.randperm(len(q),generator=vg)[:source_cfg['validation_queries_per_split']].numpy();queries.append(q[ids])
        source_rank=[numpy_ranks(Esource,R,q,cfg['batch_size']) for q in queries];after_rank=[numpy_ranks(E,R,q,cfg['batch_size']) for q in queries]
        old_before=numpy_ranks(initial['oldE'].numpy(),R,queries[0],cfg['batch_size']);protected=old_before<=10;assert protected.any()
        expected=dict(source=[metrics(r) for r in source_rank],after=[metrics(r) for r in after_rank],
                      source_old_top10_loss=float((source_rank[0][protected]>10).mean()),after_old_top10_loss=float((after_rank[0][protected]>10).mean()))
        assert expected==tr['validation'],'Independent full VALID metrics or query selection mismatch'
        from hashlib import sha256
        result=dict(status='PASS_FULL_COMMON_VALID_METRICS',failures=[],id=job['id'],method=source_job['method'],seed=job['seed'],validation=expected,
                    queries_replayed=5*len(queries[0]),old_valid_query_sha256=sha256(queries[0].tobytes()).hexdigest(),new_valid_query_sha256=sha256(queries[1].tobytes()).hexdigest(),
                    evidence_sha256={str(out/name):sha(out/name) for name in ['best.pt','initial.pt','training-receipt.json','audit.json']},
                    limits=['Complete fixed common VALID metrics replay, not independent source SGD retraining.','Development screen; neither heldout nor SOTA inference.'])
    except Exception as exc:
        (dest/'audit.json').write_text(json.dumps(dict(status='FAIL_RETAINED',failures=[repr(exc)]),indent=2));raise
    (dest/'audit.json').write_text(json.dumps(result,indent=2));print('FULL_VALID_UTILITY_AUDIT_PASS',job['id'],flush=True)


def aggregate(root,cfg):
    source_can=yaml.safe_load((root/'experiments'/cfg['source_experiment']/'experiment.yaml').read_text());source_cfg=json.loads((root/source_can['config']).read_text())
    canonical=yaml.safe_load((root/'experiments'/cfg['id']/'experiment.yaml').read_text());assert 'terminal' in canonical['status'] and not canonical['active_processes']
    groups=defaultdict(list);unavailable=[]
    for job in cfg['jobs']:
        if canonical['jobs'][job['id']]['status']!='completed_audited':unavailable.append(dict(id=job['id'],method=job['method'],seed=job['seed'],status=canonical['jobs'][job['id']]['status']));continue
        row=json.loads((root/cfg['reaudit_directory']/job['id']/'audit.json').read_text());assert row['status']=='PASS_FULL_COMMON_VALID_METRICS';groups[job['method']].append(row)
    records=[]
    for method in dict.fromkeys(j['method'] for j in cfg['jobs']):
        rows=sorted(groups[method],key=lambda r:r['seed']);seeds=[r['seed'] for r in rows]
        if seeds!=cfg['seeds']:records.append(dict(method=method,seeds=seeds,decision='INCOMPLETE_UTILITY_EVIDENCE'));continue
        source_new=np.asarray([r['validation']['source'][1]['raw_mrr'] for r in rows]);after_new=np.asarray([r['validation']['after'][1]['raw_mrr'] for r in rows])
        source_loss=np.asarray([r['validation']['source_old_top10_loss'] for r in rows]);after_loss=np.asarray([r['validation']['after_old_top10_loss'] for r in rows])
        new_gate=bool(after_new.mean()>=source_cfg['utility_new_valid_ratio']*source_new.mean())
        old_gate=bool(source_loss.mean()>0 and after_loss.mean()<=source_cfg['utility_old_valid_loss_ratio']*source_loss.mean() and np.all(after_loss<=source_loss))
        records.append(dict(method=method,seeds=seeds,decision='PASS_DEVELOPMENT_UTILITY_SCREEN' if new_gate and old_gate else 'NO_GO_FOR_THIS_FIXED_COMBINATION',new_gate=new_gate,old_gate=old_gate,
                            source_new_valid_mrr=source_new.tolist(),after_new_valid_mrr=after_new.tolist(),source_old_valid_loss=source_loss.tolist(),after_old_valid_loss=after_loss.tolist(),
                            paired_new_mrr_deltas=(after_new-source_new).tolist(),paired_old_loss_deltas=(after_loss-source_loss).tolist()))
    dest=root/'results'/cfg['id'];dest.mkdir(exist_ok=False)
    result=dict(scope='Independent complete fixed-VALID development utility screen; no heldout/SOTA or novelty claim',thresholds=dict(new_valid_ratio=source_cfg['utility_new_valid_ratio'],old_valid_loss_ratio=source_cfg['utility_old_valid_loss_ratio']),
                rows=records,unavailable=unavailable,source_experiment=cfg['source_experiment']);(dest/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job');p.add_argument('--stage',choices=['audit','aggregate'],default='audit');a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));torch.set_num_threads(2);cfg=json.loads((root/a.config).read_text())
    if a.stage=='aggregate':aggregate(root,cfg)
    else:audit(root,cfg,next(j for j in cfg['jobs'] if j['id']==a.job))
