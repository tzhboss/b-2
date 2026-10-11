"""Queue clean EXBert foundation tasks on GPUs without live peer claims."""
import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import yaml

def main(root,config):
    cfg=json.loads((root/config).read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';s=yaml.safe_load(can.read_text())
    assert s['status']=='lm_data_preparation_passed'
    for name,digest in s['frozen_input_sha256'].items():assert sha256((root/name).read_bytes()).hexdigest()==digest,name
    for name,digest in s['external_source_sha256'].items():assert sha256(Path(name).read_bytes()).hexdigest()==digest,name
    # Keep the venv executable path: resolving its symlink bypasses the venv.
    python=Path(os.path.abspath(root/cfg['python']));pending=list(cfg['jobs']);active={};free=set(cfg['available_gpu_indices']);preflight=False
    s.update(status='running_clean_lm_base_queue',coordinator_pid=os.getpid(),jobs={})
    def save():
        s['active_processes']=[dict(id=j,pid=a['proc'].pid,stage=a['stage'],gpu=a['gpu']) for j,a in active.items()]
        s['queued_jobs']=[j['id'] for j in pending]
        tmp=can.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(s,sort_keys=False));tmp.replace(can)
    def claimed_by_peers():
        claimed=set()
        for name in cfg['resource_claim_sources']:
            state=yaml.safe_load((root/name).read_text())
            for a in state.get('active_processes',[]):
                if a.get('gpu') is None:continue
                try:os.kill(a['pid'],0)
                except ProcessLookupError:continue
                claimed.add(a['gpu'])
        return claimed
    def start(job,gpu,stage):
        jid=job['id'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if stage=='audit' else str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(root/cfg['log_directory']/f'{jid}-{stage}.log').open('xb')
        cmd=[str(python),'-u',str(root/'scripts/run_lm_common_base.py'),'--root',str(root),'--config',config,'--job',jid,'--stage',stage]
        proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[jid]=dict(job=job,gpu=gpu,stage=stage,proc=proc,log=log);s['jobs'][jid]=dict(status='running',stage=stage,pid=proc.pid,gpu=gpu)
        print('LM_QUEUE_START',jid,stage,proc.pid,gpu,flush=True)
    save()
    while pending or active:
        usage=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        idle={int(i) for line in usage.splitlines() for i,m,u in [line.split(',')] if int(m)<=1024 and int(u)<=20}
        available=sorted(free&idle-claimed_by_peers())
        for job in list(pending):
            if not available:break
            if not preflight and active:break
            gpu=available.pop(0);free.remove(gpu);pending.remove(job);start(job,gpu,'train' if preflight else 'preflight')
        for jid,a in list(active.items()):
            rc=a['proc'].poll()
            if rc is None:continue
            a['log'].close();del active[jid];stage=a['stage'];gpu=a['gpu'];job=a['job']
            if rc==0 and stage=='preflight':preflight=True;s['preflight_passed']=True;start(job,gpu,'train')
            elif rc==0 and stage=='train':free.add(gpu);start(job,None,'audit')
            else:
                if stage!='audit':free.add(gpu)
                s['jobs'][jid]=dict(status='completed_audited_clean_base_only' if rc==0 else 'failed_retained',stage=stage,exit_code=rc)
                print('LM_QUEUE_TERMINAL',jid,stage,rc,flush=True)
                if rc and stage=='preflight':
                    for j in pending:s['jobs'][j['id']]=dict(status='blocked_by_retained_preflight_failure')
                    pending=[]
        save()
        if pending or active:time.sleep(3)
    successful=[]
    for job in cfg['jobs']:
        if s['jobs'][job['id']]['status']=='completed_audited_clean_base_only':
            out=root/'artifacts/lm-base'/job['id'];successful.append(dict(id=job['id'],training=json.loads((out/'terminal.json').read_text()),audit=json.loads((out/'audit.json').read_text())))
    dest=root/'results'/cfg['id'];dest.mkdir(exist_ok=False);(dest/'summary.json').write_text(json.dumps(dict(scope='Clean old foundation only; KGEditor meta and online training still required',jobs=successful),indent=2))
    s.update(status='completed_audited_clean_lm_bases_only' if len(successful)==len(cfg['jobs']) else 'lm_base_terminal_with_retained_failures',completed_utc=datetime.now(timezone.utc).isoformat(),result_directory=str(dest.relative_to(root)))
    save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();main(a.root.resolve(),a.config)
