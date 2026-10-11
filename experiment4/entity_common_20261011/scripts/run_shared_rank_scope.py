"""Config-driven TRAIN guard compilation and independent source audit CLI."""
import argparse,json,sys
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['prepare','audit'],required=True)
    a=p.parse_args();root=a.root.resolve();cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    sys.path.insert(0,str(root/'src'));from shared_rank_scope import prepare,audit
    (prepare if a.stage=='prepare' else audit)(root,cfg,job)
