"""Config-driven native meta, cumulative online, evaluation and audit worker."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job')
    p.add_argument('--stage',default='audit',choices=['prepare','cpu-smoke','preflight','train','evaluate','audit']);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));cfg=json.loads((root/a.config).read_text())
    from lm_editor_common_v2 import prepare_online,train_online,evaluate,audit
    from lm_editor_bridge_v2 import train_meta
    if a.stage=='prepare':prepare_online(root,cfg);return
    job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='cpu-smoke':train_online(root,cfg,job,cpu_smoke=True)
    elif a.stage in ['preflight','train']:
        train_meta(root,cfg,job,preflight=a.stage=='preflight');train_online(root,cfg,job,preflight=a.stage=='preflight')
    elif a.stage=='evaluate':evaluate(root,cfg,job)
    else:audit(root,cfg,job)


if __name__=='__main__':main()
