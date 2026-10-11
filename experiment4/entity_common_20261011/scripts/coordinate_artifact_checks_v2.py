"""Resource-aware artifact stages with per-job immutable source dependencies."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import yaml


def main(root,config):
    cfg=json.loads((root/config).read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';state=yaml.safe_load(can.read_text());assert state['status']=='registered'
    for name,digest in state['frozen_input_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    state.update(status='running_registered_artifact_checks',coordinator_pid=os.getpid(),jobs={},active_processes=[])
    pending=list(cfg['jobs']);audits=[];active={};free=set(cfg['available_gpu_indices']);logdir=root/cfg['log_directory'];logdir.mkdir(parents=True,exist_ok=True)
    def save():
        state['active_processes']=[dict(id=jid,pid=a['p'].pid,stage=a['stage'],gpu=a['gpu']) for jid,a in active.items()]
        state['queued_jobs']=[j['id'] for j in pending];state['queued_audits']=[j['id'] for j in audits]
        temp=can.with_suffix('.tmp');temp.write_text(yaml.safe_dump(state,sort_keys=False));temp.replace(can)
    def start(job,gpu,stage):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        log=(logdir/f"{job['id']}-{stage}.log").open('xb');python=os.path.abspath(root/cfg['python'])
        p=subprocess.Popen([python,'-u',str(root/cfg['worker']),'--root',str(root),'--config',config,'--job',job['id'],'--stage',stage],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[job['id']]=dict(job=job,p=p,log=log,stage=stage,gpu=gpu);state['jobs'][job['id']]=dict(status='running',stage=stage,pid=p.pid,gpu=gpu)
        print('ARTIFACT_START',job['id'],stage,p.pid,gpu,flush=True)
    save()
    while pending or audits or active:
        claimed=set()
        for name in cfg.get('resource_claim_sources',[]):
            peer=yaml.safe_load((root/name).read_text())
            for a in peer.get('active_processes',[]):
                if a.get('gpu') is None:continue
                try:os.kill(a['pid'],0)
                except ProcessLookupError:continue
                claimed.add(a['gpu'])
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        idle={int(i) for line in usage.splitlines() for i,m,u in [line.split(',')] if int(m)<=1024 and int(u)<=20}-claimed
        for job in list(pending):
            dependency=cfg.get('source_job_dependency')
            if dependency:
                source=yaml.safe_load((root/dependency).read_text());upstream=source.get('jobs',{}).get(job['id'],{})
                if upstream.get('status') not in ['completed_audited','failed_retained','blocked_by_retained_method_preflight_failure']:continue
            available=sorted(free&idle)
            if not available:break
            gpu=available[0];free.remove(gpu);idle.remove(gpu);pending.remove(job);start(job,gpu,'prepare')
        for jid,a in list(active.items()):
            rc=a['p'].poll()
            if rc is None:continue
            a['log'].close();del active[jid]
            if a['stage']=='prepare':free.add(a['gpu'])
            if rc==0 and a['stage']=='prepare':audits.append(a['job']);state['jobs'][jid]=dict(status='waiting_audit')
            else:state['jobs'][jid]=dict(status='completed_audited' if rc==0 else 'failed_retained',stage=a['stage'],exit_code=rc)
            print('ARTIFACT_STAGE_TERMINAL',jid,a['stage'],rc,flush=True)
        while audits and sum(a['stage']=='audit' for a in active.values())<cfg['max_concurrent_audits']:start(audits.pop(0),None,'audit')
        save()
        if pending or audits or active:time.sleep(3)
    rows=[]
    for job in cfg['jobs']:
        if state['jobs'][job['id']]['status']=='completed_audited':
            out=root/cfg['output_directory']/job['id'];rows.append(dict(result=json.loads((out/'receipt.json').read_text()),audit=json.loads((out/'audit.json').read_text())))
    dest=root/'results'/cfg['id'];dest.mkdir(parents=True,exist_ok=False);(dest/'summary.json').write_text(json.dumps(dict(scope=cfg['scope'],jobs_passed=len(rows),jobs_total=len(cfg['jobs']),rows=rows),indent=2))
    state.update(status='completed_audited_artifact_checks_only' if len(rows)==len(cfg['jobs']) else 'terminal_artifact_checks_with_retained_failures',result_directory=str(dest.relative_to(root)))
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();main(a.root.resolve(),a.config)
