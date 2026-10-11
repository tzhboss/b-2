"""Run config-driven native language-model numerical diagnostics."""
import argparse,json,sys
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'))
    from lm_numerical_replay import inspect
    print(json.dumps(inspect(root,json.loads((root/a.config).read_text())),indent=2))
