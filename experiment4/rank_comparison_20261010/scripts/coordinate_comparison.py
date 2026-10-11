"""Canonical status and concurrent bounded CPU comparison jobs."""
import argparse, concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
import yaml
from run_rank_comparison import sha,verify

def main(root):
    canon=root/'experiments/KGE-RANK-COMP-001/experiment.yaml'
    state=yaml.safe_load(canon.read_text());assert state['status']=='registered'
    verify(root);state.update(status='running',coordinator_pid=os.getpid(),active_processes=[])
    def save():
        tmp=canon.with_suffix('.yaml.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(canon)
    save(); env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    processes={}
    for model in ['distmult','complex']:
        log=(root/'logs'/f'{model}-replay.log').open('xb')
        command=[sys.executable,'-u',str(root/'scripts/run_rank_comparison.py'),'--root',str(root),'--model',model]
        proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes[model]=dict(proc=proc,log=log,stage='replay')
    while processes:
        for model,a in list(processes.items()):
            rc=a['proc'].poll()
            if rc is None:continue
            a['log'].close()
            if rc==0 and a['stage']=='replay':
                log=(root/'logs'/f'{model}-audit.log').open('xb')
                proc=subprocess.Popen([sys.executable,'-u',str(root/'scripts/audit_rank_comparison.py'),'--root',str(root),'--model',model],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                processes[model]=dict(proc=proc,log=log,stage='audit');continue
            state['jobs'][model]=dict(stage=a['stage'],exit_code=rc,status='audited' if rc==0 else 'failed_retained')
            del processes[model]
        state['active_processes']=[dict(model=m,pid=a['proc'].pid,stage=a['stage']) for m,a in processes.items()];save()
        if processes:time.sleep(2)
    state['status']='terminal_audited_pending_controller_review' if all(j['status']=='audited' for j in state['jobs'].values()) else 'failed_retained'
    save();print(state['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
