"""Registered resource-aware base training queue with sequential CPU audit."""
import argparse,json,os,subprocess,time
from pathlib import Path
import yaml
from hashlib import sha256
def main(root):
    cfg=json.loads((root/'configs/base-training.json').read_text());canonical=root/'experiments/KGE-ENTITY-COMMON-BASE-001/experiment.yaml'
    state=yaml.safe_load(canonical.read_text());assert state['status']=='registered'
    for name,d in state['frozen_input_sha256'].items():assert sha256((root/name).read_bytes()).hexdigest()==d,name
    python=(root/cfg['python']).resolve();free=set(cfg['available_gpu_indices']);pending=list(cfg['jobs']);active={}
    state.update(status='running_base_training',coordinator_pid=os.getpid(),jobs={},active_processes=[],method_comparison_started=False)
    def save():
        tmp=canonical.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(canonical)
    save()
    while pending or active:
        for job in list(pending):
            if not free:break
            gpu=min(free);free.remove(gpu);pending.remove(job);jid=job['id']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            log=(root/'logs/base'/f'{jid}-train.log').open('xb')
            proc=subprocess.Popen([str(python),'-u',str(root/'scripts/train_common_base.py'),'--root',str(root),'--job',jid],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[jid]=dict(process=proc,log=log,env=env,gpu=gpu,stage='train');print('START_BASE',jid,proc.pid,gpu,flush=True)
        for jid,a in list(active.items()):
            rc=a['process'].poll()
            if rc is None:continue
            a['log'].close()
            if a['stage']=='train':free.add(a['gpu'])
            if rc==0 and a['stage']=='train':
                log=(root/'logs/base'/f'{jid}-audit.log').open('xb');env=dict(a['env'],CUDA_VISIBLE_DEVICES='')
                proc=subprocess.Popen([str(python),'-u',str(root/'scripts/audit_common_base.py'),'--root',str(root),'--job',jid],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                a.update(process=proc,log=log,stage='audit');continue
            state['jobs'][jid]=dict(status='completed_audited_pending_controller' if rc==0 else 'failed_retained',stage=a['stage'],exit_code=rc)
            del active[jid];print('TERMINAL_BASE',jid,rc,flush=True)
        state['active_processes']=[dict(id=jid,pid=a['process'].pid,stage=a['stage'],gpu=a['gpu'] if a['stage']=='train' else None) for jid,a in active.items()]
        state['queued_jobs']=[j['id'] for j in pending];save()
        if active or pending:time.sleep(3)
    state.update(status='all_base_terminal_pending_controller_review',active_processes=[],queued_jobs=[]);save()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
