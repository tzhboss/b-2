import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job');p.add_argument('--seed',type=int);p.add_argument('--stage',choices=['features','preflight','train','evaluate','audit'],required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));cfg=json.loads((root/a.config).read_text())
    from rank_entity import prepare_features,train_rank,audit_rank_steps
    from common_incremental import evaluate
    if a.stage=='features':prepare_features(root,cfg,a.seed);return
    job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='evaluate':evaluate(root,cfg,job)
    elif a.stage=='audit':audit_rank_steps(root,cfg,job)
    else:train_rank(root,cfg,job,preflight=a.stage=='preflight')


if __name__=='__main__':main()
