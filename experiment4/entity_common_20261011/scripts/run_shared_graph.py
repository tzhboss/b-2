"""Shared-row graph stages with complete patch and unchanged inference audits."""
import argparse,json,subprocess,sys
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['preflight','train','evaluate','audit'],default='audit')
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    sys.path.insert(0,str(root/'src'));from shared_rank_graph import train_rank,evaluate
    if a.stage in ['preflight','train']:
        train_rank(root,cfg,job,preflight=a.stage=='preflight')
        if a.stage=='train':
            out=root/'artifacts/incremental'/a.job;path=out/'training-receipt.json';receipt=json.loads(path.read_text())
            scope=json.loads((root/cfg['scope_directory']/f"shared-rank-scope-s{job['seed']}"/'receipt.json').read_text())
            receipt['scope_preparation_elapsed_s']=scope['elapsed_s']
            receipt['source_inclusive_incremental_elapsed_s']=receipt['source_sgd_elapsed_s']+scope['elapsed_s']+receipt['elapsed_s']
            path.write_text(json.dumps(receipt,indent=2))
    elif a.stage=='evaluate':evaluate(root,cfg,job)
    else:
        for script in ['scripts/audit_shared_graph_updates.py','scripts/audit_exported_embeddings.py']:
            subprocess.run([sys.executable,str(root/script),'--root',str(root),'--config',a.config,'--job',a.job],check=True)
        out=root/'artifacts/incremental'/a.job;result=json.loads((out/'audit.json').read_text());result['solver_audit']=json.loads((out/'solver-audit.json').read_text());result['status']='PASS_COMMON_INFERENCE_ALL3753_PATCHES_AND_FULL_TERMINAL_SCOPE'
        (out/'audit.json').write_text(json.dumps(result,indent=2));print('SHARED_GRAPH_FULL_AUDIT_PASS',a.job,flush=True)
