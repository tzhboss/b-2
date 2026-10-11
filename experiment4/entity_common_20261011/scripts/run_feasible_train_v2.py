"""Retain040 arithmetic/audits, replace only infeasible immutable working rows."""
import argparse,json,sys
from pathlib import Path
import run_feasible_train40 as original
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common_training import sha
from feasible_train_optimization_v2 import optimize_train_row


def worker(root,cfg,job):
    original.optimize_train_row=optimize_train_row
    original.worker(root,cfg,job)
    out=root/cfg['output_directory']/job['id'];p=out/'receipt.json';receipt=json.loads(p.read_text())
    receipt['optimization_module']='src/feasible_train_optimization_v2.py'
    receipt['optimization_module_sha256']=sha(root/receipt['optimization_module'])
    receipt['wrapper_sha256']=sha(Path(__file__))
    receipt['input_sha256'].update({str(root/name):sha(root/name) for name in [receipt['optimization_module'],'scripts/run_feasible_train_v2.py']})
    receipt['limits'].append('Original immutable-only tied score working-set rows omitted; actual full rank acceptance unchanged.')
    p.write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['run','audit'],required=True);p.add_argument('--job',required=True)
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='run':worker(root,cfg,job)
    else:original.audit(root,cfg,job)
