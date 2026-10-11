"""Version-configured novel-entity Rank worker, including the audit stages."""
import argparse,importlib,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);p.add_argument('--stage',choices=['preflight','train','evaluate','audit'],default='audit');a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));cfg=json.loads((root/a.config).read_text());implementation=importlib.import_module(cfg['implementation_module'])
    job=next(j for j in cfg['jobs'] if j['id']==a.job)
    if a.stage=='evaluate':implementation.evaluate(root,cfg,job)
    elif a.stage=='audit':
        implementation.audit_rank_steps(root,cfg,job)
        subprocess.run([sys.executable,str(root/'scripts/audit_exported_embeddings.py'),'--root',str(root),'--config',a.config,'--job',a.job],check=True)
        out=root/'artifacts/incremental'/a.job;audit=json.loads((out/'audit.json').read_text());audit['solver_audit']=json.loads((out/'solver-audit.json').read_text());audit['status']='PASS_COMMON_INFERENCE_AND_ALL_STEP_SOLVER_REPLAY'
        (out/'audit.json').write_text(json.dumps(audit,indent=2));print('FULL_RANK_AUDIT_PASS',a.job)
    else:implementation.train_rank(root,cfg,job,preflight=a.stage=='preflight')


if __name__=='__main__':main()
