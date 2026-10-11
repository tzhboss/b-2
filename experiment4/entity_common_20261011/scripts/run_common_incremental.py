import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
p.add_argument('--stage',choices=['preflight','train','evaluate'],required=True);a=p.parse_args()
root=a.root.resolve();sys.path.insert(0,str(root/'src'))
from common_incremental import train,evaluate
cfg=json.loads((root/'configs/incremental-training.json').read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
if a.stage=='evaluate':evaluate(root,cfg,job)
else:train(root,cfg,job,preflight=a.stage=='preflight')
