"""Native editor feasibility entrypoint; whole online stage is separately required."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job');p.add_argument('--stage',choices=['prepare','meta-smoke'],required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));cfg=json.loads((root/a.config).read_text())
    from lm_editor_bridge_v2 import prepare_meta,train_meta
    if a.stage=='prepare':prepare_meta(root,cfg)
    else:train_meta(root,cfg,next(j for j in cfg['jobs'] if j['id']==a.job),cpu_smoke=True)


if __name__=='__main__':main()
