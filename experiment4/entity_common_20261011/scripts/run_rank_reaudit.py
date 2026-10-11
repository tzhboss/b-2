"""Re-audit unchanged original exports, preserving original failure evidence."""
import argparse,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job',required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));cfg=json.loads((root/a.config).read_text());job=next(j for j in cfg['jobs'] if j['id']==a.job)
    dest=root/cfg['reaudit_directory']/job['id'];dest.mkdir(parents=True,exist_ok=False)
    from rank_entity_reaudit import audit_rank_steps
    try:
        audit_rank_steps(root,cfg,job)
        subprocess.run([sys.executable,str(root/'scripts/audit_exported_embeddings_v2.py'),'--root',str(root),'--config',a.config,'--job',a.job],check=True)
        result=json.loads((dest/'inference-audit.json').read_text());result['solver_audit']=json.loads((dest/'solver-audit.json').read_text())
        assert not result['failures'] and result['solver_audit']['status']=='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS'
        result['status']='PASS_COMMON_INFERENCE_AND_CORRECTED_SOLVER_REAUDIT'
    except Exception as exc:
        (dest/'audit.json').write_text(json.dumps(dict(status='FAIL_RETAINED',failures=[repr(exc)]),indent=2));raise
    (dest/'audit.json').write_text(json.dumps(result,indent=2));print('CORRECTED_RAW_REAUDIT_PASS',a.job,flush=True)


if __name__=='__main__':main()
