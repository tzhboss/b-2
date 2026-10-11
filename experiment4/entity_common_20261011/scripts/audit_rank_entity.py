"""Require the full-step solver audit and common TEST inference audit together."""
import argparse,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);a=p.parse_args();root=a.root.resolve()
    args=['--root',str(root),'--config',a.config,'--job',a.job]
    subprocess.run([sys.executable,str(root/'scripts/run_rank_entity.py'),*args,'--stage','audit'],check=True)
    subprocess.run([sys.executable,str(root/'scripts/audit_exported_embeddings.py'),*args],check=True)
    out=root/'artifacts/incremental'/a.job;audit=json.loads((out/'audit.json').read_text());audit['solver_audit']=json.loads((out/'solver-audit.json').read_text())
    assert audit['solver_audit']['status']=='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS'
    audit['status']='PASS_COMMON_INFERENCE_AND_ALL_STEP_SOLVER_REPLAY';(out/'audit.json').write_text(json.dumps(audit,indent=2));print('FULL_RANK_AUDIT_PASS',a.job)


if __name__=='__main__':main()
