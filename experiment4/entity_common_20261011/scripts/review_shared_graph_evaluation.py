"""Evaluate and audit unchanged trained models after a configuration failure."""
import argparse,json,subprocess,sys
from pathlib import Path
from hashlib import sha256
import yaml


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['prepare','audit'],required=True);a=p.parse_args()
    root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    assert sha256((root/cfg['source_config']).read_bytes()).hexdigest()==cfg['source_config_sha256']
    original=json.loads((root/cfg['source_config']).read_text());source_job=next(j for j in original['jobs'] if j['id']==a.job)
    assert job==source_job
    for key in ['source_template','verified_proposal_template','parameter_budget','solver','validation_seed','validation_queries_per_split',
                'utility_new_mrr_ratio','utility_old_loss_ratio','utility_direction_mrr_ratio','utility_direction_hits10_ratio']:assert cfg[key]==original[key],key
    source=yaml.safe_load((root/cfg['source_job_dependency']).read_text());state=source['jobs'][a.job]
    assert state['status']=='failed_retained' and state['stage']=='evaluate' and state['exit_code']==1
    failed_log=root/original['log_directory']/f'{a.job}-evaluate.log';assert "KeyError: 'cpu_threads_per_job'" in failed_log.read_text()
    out=root/'artifacts/incremental'/a.job;receipt=json.loads((out/'training-receipt.json').read_text());model_hash=sha256((out/'best.pt').read_bytes()).hexdigest()
    assert receipt['processed_new_entities']==3753 and not receipt['test_read'] and model_hash==receipt['checkpoint_sha256']
    dest=root/cfg['output_directory']/a.job;sys.path.insert(0,str(root/'src'))
    if a.stage=='prepare':
        dest.mkdir(parents=True,exist_ok=False)
        assert not (out/'evaluation.json').exists() and not (out/'ranks.npz').exists(),'Unexpected prior evaluation; preserve rather than overwrite'
        from common_incremental import evaluate
        evaluate(root,cfg,job)
        result=dict(id=a.job,scope='Unchanged027 weights, evaluation-only missing thread configuration repair',checkpoint_sha256=model_hash,
                    training_receipt_sha256=sha256((out/'training-receipt.json').read_bytes()).hexdigest(),source_failed_log_sha256=sha256(failed_log.read_bytes()).hexdigest(),
                    evaluation=json.loads((out/'evaluation.json').read_text()),retrained=False,source_config_sha256=cfg['source_config_sha256'])
        (dest/'receipt.json').write_text(json.dumps(result,indent=2))
    else:
        for script in ['scripts/audit_shared_graph_updates.py','scripts/audit_exported_embeddings.py']:
            subprocess.run([sys.executable,str(root/script),'--root',str(root),'--config',a.config,'--job',a.job],check=True)
        result=json.loads((out/'audit.json').read_text());result['solver_audit']=json.loads((out/'solver-audit.json').read_text())
        result.update(status='PASS_COMMON_INFERENCE_ALL3753_PATCHES_AND_FULL_TERMINAL_SCOPE',retrained=False,evaluation_only_config_repair=True,
                      source_checkpoint_sha256=model_hash,source_config_sha256=cfg['source_config_sha256'])
        (out/'audit.json').write_text(json.dumps(result,indent=2));(dest/'audit.json').write_text(json.dumps(result,indent=2))
    assert sha256((out/'best.pt').read_bytes()).hexdigest()==model_hash
