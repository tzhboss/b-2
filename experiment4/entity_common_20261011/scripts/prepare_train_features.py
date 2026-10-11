import argparse,json,os,sys
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
root=a.root.resolve();cfg=json.loads((root/a.config).read_text());sys.path.insert(0,str(root/'src'))
from train_graph_features import generate
canonical=root/'experiments'/cfg['id']/'experiment.yaml';state=yaml.safe_load(canonical.read_text())
assert state['status']=='registered_train_feature_preparation'
state.update(status='running_train_feature_preparation',active_processes=[dict(pid=os.getpid(),stage='exact TRAIN centrality',gpu=None)])
canonical.write_text(yaml.safe_dump(state,sort_keys=False))
try:
    generate(root,cfg)
except Exception:
    state.update(status='train_feature_failure_retained',active_processes=[]);canonical.write_text(yaml.safe_dump(state,sort_keys=False));raise
state.update(status='train_feature_preparation_passed',active_processes=[],feature_receipt=cfg['feature_directory']+'/receipt.json')
canonical.write_text(yaml.safe_dump(state,sort_keys=False))
