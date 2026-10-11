"""Generic registered evaluation scheduler: GPU scoring followed by CPU audit."""
import argparse,json,os,subprocess,time
from pathlib import Path
import yaml

def main(root,base):
    config=json.loads((root/'config.json').read_text());canonical=root/'experiments/KGE-RANK-EVAL-001/experiment.yaml'
    state=yaml.safe_load(canonical.read_text());assert state['status']=='registered'
    pending=list(config['jobs']);active={};free=set(range(7));state.update(status='running',coordinator_pid=os.getpid(),jobs={},active_processes=[])
    def save():
        tmp=canonical.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(canonical)
    while pending or active:
        for job in list(pending):
            if not free:break
            gpu=min(free);free.remove(gpu);pending.remove(job);jid=job['id']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            py=base/job['python'];log=(root/'logs'/f'{jid}-score.log').open('xb')
            p=subprocess.Popen([str(py),'-u',str(root/'scripts'/job['runner']),'--root',str(root),'--base',str(base),'--job',jid],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[jid]=dict(job=job,p=p,log=log,gpu=gpu,env=env,stage='score');print('START',jid,p.pid,gpu,flush=True)
        for jid,a in list(active.items()):
            rc=a['p'].poll()
            if rc is None:continue
            a['log'].close()
            if a['stage']=='score':free.add(a['gpu'])
            auditor=a['job'].get('auditor')
            if rc==0 and a['stage']=='score' and auditor:
                log=(root/'logs'/f'{jid}-audit.log').open('xb')
                py=base/'environment/downloads/bin/python';env=dict(a['env'],CUDA_VISIBLE_DEVICES='')
                p=subprocess.Popen([str(py),'-u',str(root/'scripts'/auditor),'--root',str(root),'--base',str(base),'--job',jid],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                a.update(p=p,log=log,stage='audit');continue
            state['jobs'][jid]=dict(status='terminal_pending_controller_review' if rc==0 else 'failed_retained',stage=a['stage'],exit_code=rc)
            del active[jid];print('TERMINAL',jid,rc,flush=True)
        state['active_processes']=[dict(id=jid,pid=a['p'].pid,stage=a['stage'],gpu=a['gpu'] if a['stage']=='score' else None) for jid,a in active.items()];save()
        if pending or active:time.sleep(3)
    state['status']='all_terminal_pending_controller_review';save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.base.resolve())
