"""Config-driven training queues with source pins, GPU dependencies and audits."""
import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import numpy as np
import yaml

def main(root,config):
    cfg=json.loads((root/config).read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';s=yaml.safe_load(can.read_text())
    assert s['status']=='registered'
    for name,digest in s['frozen_input_sha256'].items():assert sha256((root/name).read_bytes()).hexdigest()==digest,name
    for name,digest in s['external_source_sha256'].items():assert sha256(Path(name).read_bytes()).hexdigest()==digest,name
    pending=list(cfg['jobs']);pending_audits=[];active={};free=set(cfg['available_gpu_indices']);preflight_passed=set()
    def preflight_group(job):return job['id'] if cfg.get('preflight_per_job',False) else job['method']
    s.update(status='waiting_for_gpu_queue_dependency',coordinator_pid=os.getpid(),jobs={},preflight_passed=[],active_processes=[])
    def save():
        s['active_processes']=[dict(id=j,pid=a['proc'].pid,stage=a['stage'],gpu=a['gpu']) for j,a in active.items()]
        s['queued_jobs']=[j['id'] for j in pending];s['preflight_passed']=sorted(preflight_passed)
        s['queued_audits']=[j['id'] for j in pending_audits]
        tmp=can.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(s,sort_keys=False));tmp.replace(can)
    def dependencies_done():
        for path in cfg.get('gpu_queue_dependencies',[]):
            d=yaml.safe_load((root/path).read_text())
            if d.get('active_processes') or d.get('queued_jobs'):return False
            if 'terminal' not in d['status'] and 'completed' not in d['status']:return False
            required=cfg.get('required_dependency_statuses',{}).get(path)
            if required is not None and d['status']!=required:return False
        return True
    def claimed_by_peers():
        claimed=set()
        for name in cfg.get('resource_claim_sources',[]):
            peer=yaml.safe_load((root/name).read_text())
            for a in peer.get('active_processes',[]):
                if a.get('gpu') is None:continue
                try:os.kill(a['pid'],0)
                except ProcessLookupError:continue
                claimed.add(a['gpu'])
        return claimed
    def peer_audits():
        count=0
        for name in cfg.get('resource_claim_sources',[]):
            peer=yaml.safe_load((root/name).read_text())
            for a in peer.get('active_processes',[]):
                if a.get('stage')!='audit':continue
                try:os.kill(a['pid'],0)
                except ProcessLookupError:continue
                count+=1
        return count
    def start(job,gpu,stage):
        jid=job['id'];python=Path(os.path.abspath(root/job.get('python',cfg['python'])))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if stage=='audit' else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(root/cfg['log_directory']/f'{jid}-{stage}.log').open('xb')
        worker=job.get('auditor',cfg['auditor']) if stage=='audit' else job.get('worker',cfg['worker'])
        cmd=[str(python),'-u',str(root/worker),'--root',str(root),'--config',config,'--job',jid]
        if stage!='audit':cmd+=['--stage',stage]
        proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[jid]=dict(job=job,gpu=gpu,stage=stage,proc=proc,log=log)
        s['jobs'][jid]=dict(status='running',stage=stage,pid=proc.pid,gpu=gpu)
        print('QUEUE_START',jid,stage,proc.pid,gpu,flush=True)
    save()
    while pending or active or pending_audits:
        if dependencies_done():
            s['status']='running_registered_comparison'
            usage=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
            idle={int(i) for line in usage.splitlines() for i,m,u in [line.split(',')] if int(m)<=1024 and int(u)<=20}-claimed_by_peers()
            for job in list(pending):
                available=sorted(free&idle)
                if not available:break
                method=preflight_group(job);needs_preflight=method not in preflight_passed
                if needs_preflight and any(preflight_group(a['job'])==method for a in active.values()):continue
                gpu=available[0];free.remove(gpu);idle.remove(gpu);pending.remove(job)
                start(job,gpu,'preflight' if needs_preflight else 'train')
        for jid,a in list(active.items()):
            rc=a['proc'].poll()
            if rc is None:continue
            a['log'].close();del active[jid];job=a['job'];gpu=a['gpu'];stage=a['stage']
            if rc==0 and stage=='preflight':preflight_passed.add(preflight_group(job));start(job,gpu,'train')
            elif rc==0 and stage=='train':start(job,gpu,'evaluate')
            elif rc==0 and stage=='evaluate':
                free.add(gpu);pending_audits.append(job);s['jobs'][jid]=dict(status='waiting_audit')
            else:
                if stage!='audit':free.add(gpu)
                s['jobs'][jid]=dict(status='completed_audited' if rc==0 else 'failed_retained',stage=stage,exit_code=rc)
                if rc and stage=='preflight':
                    for other in list(pending):
                        if preflight_group(other)==preflight_group(job):
                            pending.remove(other);s['jobs'][other['id']]=dict(status='blocked_by_retained_method_preflight_failure',source_job=jid)
                print('QUEUE_TERMINAL',jid,stage,rc,flush=True)
        available_audits=max(0,cfg.get('max_concurrent_audits',6)-peer_audits())
        while pending_audits and sum(a['stage']=='audit' for a in active.values())<available_audits:
            start(pending_audits.pop(0),None,'audit')
        save()
        if pending or active or pending_audits:time.sleep(3)
    rows=[];audits=[]
    for job in cfg['jobs']:
        if s['jobs'][job['id']]['status']!='completed_audited':continue
        out=root/'artifacts/incremental'/job['id'];r=json.loads((out/'evaluation.json').read_text())
        r['training']=json.loads((out/'training-receipt.json').read_text());rows.append(r)
        audits.append(dict(id=job['id'],**json.loads((out/'audit.json').read_text())))
    dest=root/'results'/cfg['id'];dest.mkdir(exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(dict(scope=cfg['scope'],rows=rows),indent=2))
    (dest/'audit-summary.json').write_text(json.dumps(dict(jobs_total=len(cfg['jobs']),jobs_passed=len(audits),audits=audits),indent=2))
    groups={}
    for r in rows:groups.setdefault((r['method'],r['kind']),[]).append(r)
    aggregate=[]
    for (method,kind),rs in groups.items():
        values=dict(old_mrr=[r['old_after']['raw_mrr'] for r in rs],new_mrr=[r['new_after']['raw_mrr'] for r in rs],old_top10_loss=[r['old_top10_loss'] for r in rs],new_hits10=[r['new_after']['hits10'] for r in rs])
        aggregate.append(dict(method=method,kind=kind,seeds=[r['seed'] for r in rs],metrics={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else None) for k,v in values.items()}))
    (dest/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    s.update(status='batch_terminal_audited_full_comparison_pending' if len(audits)==len(cfg['jobs']) else 'batch_terminal_with_retained_failures',
             completed_utc=datetime.now(timezone.utc).isoformat(),result_directory=str(dest.relative_to(root)))
    save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();main(a.root.resolve(),a.config)
