import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common_training import train_old
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args();root=a.root.resolve()
    cfg=json.loads((root/'configs/base-training.json').read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job);train_old(root,cfg,job)
