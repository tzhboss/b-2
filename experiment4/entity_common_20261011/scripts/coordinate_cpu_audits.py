"""Resource-aware queue for separate audits of unchanged artifacts."""
import argparse,json,os,subprocess,time
from pathlib import Path
from hashlib import sha256
import yaml


def main(root,config):
    cfg=json.loads((root/config).read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';s=yaml.safe_load(can.read_text());assert s['status']=='registered'
    for name,digest in s['frozen_input_sha256'].items():assert sha256((root/name).read_bytes()).hexdigest()==digest
    pending=list(cfg['jobs']);active={};s.update(status='waiting_resource_audit_queue',coordinator_pid=os.getpid(),jobs={})
    def save():
        s['active_processes']=[dict(id=jid,stage='audit',gpu=None,pid=a['proc'].pid) for jid,a in active.items()];s['queued_jobs']=[j['id'] for j in pending]
        tmp=can.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(s,sort_keys=False));tmp.replace(can)
    save()
    while pending or active:
        dependencies=True
        for path in cfg.get('dependencies',[]):
            d=yaml.safe_load((root/path).read_text());dependencies=dependencies and not d.get('active_processes') and ('terminal' in d['status'] or 'completed' in d['status'])
        peer_count=0
        for path in cfg.get('resource_claim_sources',[]):
            d=yaml.safe_load((root/path).read_text())
            for a in d.get('active_processes',[]):
                if a.get('stage')!='audit':continue
                try:os.kill(a['pid'],0)
                except ProcessLookupError:continue
                peer_count+=1
        if dependencies:
            s['status']='running_corrected_cpu_audits'
            while pending and len(active)+peer_count<cfg['max_concurrent_audits']:
                j=pending.pop(0);jid=j['id'];log=(root/cfg['log_directory']/f'{jid}.log').open('xb')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                proc=subprocess.Popen([os.path.abspath(root/cfg['python']),'-u',str(root/cfg['worker']),'--root',str(root),'--config',config,'--job',jid],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[jid]=dict(proc=proc,log=log);s['jobs'][jid]=dict(status='running',stage='audit',pid=proc.pid,gpu=None);print('REAUDIT_START',jid,proc.pid,flush=True)
        for jid,a in list(active.items()):
            rc=a['proc'].poll()
            if rc is None:continue
            a['log'].close();del active[jid];s['jobs'][jid]=dict(status='completed_audited' if rc==0 else 'failed_retained',exit_code=rc,stage='audit');print('REAUDIT_TERMINAL',jid,rc,flush=True)
        save()
        if pending or active:time.sleep(3)
    s['status']='batch_terminal_reaudit_original_records_unchanged';save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();main(a.root.resolve(),a.config)
