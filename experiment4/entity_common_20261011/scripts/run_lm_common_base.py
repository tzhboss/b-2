import argparse,json,sys,os
from pathlib import Path
import yaml
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job');p.add_argument('--stage',choices=['prepare','preflight','train','audit'],required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));cfg=json.loads((root/a.config).read_text())
    from lm_common_base import prepare,train_base,audit_base
    if a.stage=='prepare':
        can=root/'experiments'/cfg['id']/'experiment.yaml';s=yaml.safe_load(can.read_text());assert s['status']=='registered_lm_data_preparation'
        s.update(status='running_lm_data_preparation',active_processes=[dict(pid=os.getpid(),stage='old TRAIN text encoding',gpu=None)]);can.write_text(yaml.safe_dump(s,sort_keys=False))
        try:prepare(root,cfg)
        except Exception:
            s.update(status='lm_data_preparation_failure_retained',active_processes=[]);can.write_text(yaml.safe_dump(s,sort_keys=False));raise
        s.update(status='lm_data_preparation_passed',active_processes=[]);can.write_text(yaml.safe_dump(s,sort_keys=False))
        return
    job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='audit':audit_base(root,cfg,job)
    else:train_base(root,cfg,job,preflight=a.stage=='preflight')
if __name__=='__main__':main()
