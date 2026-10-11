"""Reuse registered graph stages and account for configured upstream preparation."""
import argparse, json, subprocess, sys
from pathlib import Path


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True)
    p.add_argument('--job',required=True);p.add_argument('--stage',choices=['preflight','train','evaluate','audit'],default='audit')
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    subprocess.run([sys.executable,str(root/'scripts/run_shared_graph.py'),'--root',str(root),'--config',a.config,'--job',a.job,'--stage',a.stage],check=True)
    if a.stage=='train':
        sys.path.insert(0,str(root/'src'));from common_training import sha
        out=root/'artifacts/incremental'/job['id'];receipt=json.loads((out/'training-receipt.json').read_text())
        components=[];scope_cost=receipt['scope_preparation_elapsed_s']
        components.append(dict(source='Current scope policy preparation',elapsed_s=scope_cost,amortization=1))
        for source in cfg['additional_preparation_costs']:
            path=root/source['path'].format(seed=job['seed']);record=json.loads(path.read_text());seconds=record[source['elapsed_field']]/source['amortization']
            assert seconds>=0 and source['amortization']>=1;scope_cost+=seconds
            components.append(dict(source=str(path.relative_to(root)),elapsed_s=seconds,amortization=source['amortization']))
            receipt['bound_input_sha256'][str(path)]=sha(path)
        receipt['scope_preparation_elapsed_s']=scope_cost;receipt['scope_preparation_cost_components']=components
        receipt['source_inclusive_incremental_elapsed_s']=receipt['source_sgd_elapsed_s']+scope_cost+receipt['elapsed_s']
        receipt['preparation_cost_limits']=['Shared three-seed diagnostic costs divided equally across seeds; not independently repeated per run.',
                                           'Separate global split-integrity setup is untimed and excluded, explicitly disclosed.',
                                           'Numerical solver and common scoring routines unchanged.']
        receipt['limits'].extend(cfg.get('additional_method_limits', []))
        (out/'training-receipt.json').write_text(json.dumps(receipt,indent=2))
