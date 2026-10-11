"""Config-driven clean foundation replay at registered precision."""
import argparse,json,sys
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'))
    from lm_base_reaudit import audit
    cfg=json.loads((root/a.config).read_text())
    for job in cfg['jobs']:audit(root,cfg,job)
