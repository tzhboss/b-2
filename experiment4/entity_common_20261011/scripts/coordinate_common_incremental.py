"""Bounded GPU queue: selected checkpoint, full TEST evaluation, CPU audit."""
import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import numpy as np
import yaml

def main(root):
    cfg=json.loads((root/'configs/incremental-training.json').read_text())
    canonical=root/'experiments'/cfg['id']/'experiment.yaml';state=yaml.safe_load(canonical.read_text())
    assert state['status']=='registered'
    for name,digest in state['frozen_input_sha256'].items():assert sha256((root/name).read_bytes()).hexdigest()==digest,name
    for name,digest in state['external_source_sha256'].items():assert sha256(Path(name).read_bytes()).hexdigest()==digest,name
    python=(root/cfg['python']).resolve();pending=list(cfg['jobs']);active={};free=set(cfg['available_gpu_indices'])
    state.update(status='running_incremental_comparison_first_batch',coordinator_pid=os.getpid(),jobs={},model_comparison_started=True)
    def save():
        state['active_processes']=[dict(id=j,pid=a['proc'].pid,stage=a['stage'],gpu=a['gpu']) for j,a in active.items()]
        state['queued_jobs']=[j['id'] for j in pending]
        tmp=canonical.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(canonical)
    def start(job,gpu,stage):
        jid=job['id'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if stage!='audit' else '',
           OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(root/'logs/incremental'/f'{jid}-{stage}.log').open('xb')
        cmd=[str(python),'-u',str(root/'scripts'/('audit_common_incremental.py' if stage=='audit' else 'run_common_incremental.py')),'--root',str(root),'--job',jid]
        if stage!='audit':cmd+=['--stage',stage]
        proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[jid]=dict(job=job,proc=proc,gpu=gpu,stage=stage,log=log)
        state['jobs'][jid]=dict(status='running',stage=stage,pid=proc.pid,gpu=gpu)
        print('START',jid,stage,proc.pid,gpu,flush=True)
    save()
    while pending or active:
        if free and pending:
            usage=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
            idle={int(i) for line in usage.splitlines() for i,m in [line.split(',')] if int(m)<=512}
            for gpu in sorted(free & idle):
                if not pending:break
                free.remove(gpu);start(pending.pop(0),gpu,'train')
        for jid,a in list(active.items()):
            rc=a['proc'].poll()
            if rc is None:continue
            a['log'].close();stage=a['stage'];gpu=a['gpu'];job=a['job'];del active[jid]
            if rc==0 and stage=='train':start(job,gpu,'evaluate')
            elif rc==0 and stage=='evaluate':
                free.add(gpu);start(job,None,'audit')
            else:
                if stage!='audit':free.add(gpu)
                state['jobs'][jid]=dict(status='completed_audited' if rc==0 else 'failed_retained',stage=stage,exit_code=rc)
                print('TERMINAL',jid,stage,rc,flush=True)
        save()
        if pending or active:time.sleep(3)
    successful=[];audits=[]
    for j in cfg['jobs']:
        if state['jobs'][j['id']]['status']!='completed_audited':continue
        out=root/'artifacts/incremental'/j['id'];metrics=json.loads((out/'evaluation.json').read_text())
        metrics['training']=json.loads((out/'training-receipt.json').read_text())
        successful.append(metrics);audits.append(dict(id=j['id'],**json.loads((out/'audit.json').read_text())))
    target=root/'results'/cfg['id'];target.mkdir(exist_ok=False)
    (target/'summary.json').write_text(json.dumps(dict(scope='First batch only; full comparison remains incomplete',rows=successful),indent=2))
    (target/'audit-summary.json').write_text(json.dumps(dict(jobs_passed=len(audits),jobs_total=len(cfg['jobs']),audits=audits),indent=2))
    groups={}
    for row in successful:groups.setdefault((row['method'],row['kind']),[]).append(row)
    aggregate=[]
    for (method,kind),rows in groups.items():
        values=dict(old_mrr=[r['old_after']['raw_mrr'] for r in rows],new_mrr=[r['new_after']['raw_mrr'] for r in rows],
                    old_top10_loss=[r['old_top10_loss'] for r in rows],new_hits10=[r['new_after']['hits10'] for r in rows])
        aggregate.append(dict(method=method,kind=kind,seeds=[r['seed'] for r in rows],
                    metrics={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else None) for k,v in values.items()}))
    (target/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    state.update(status='first_batch_terminal_audited_remaining_methods_pending' if len(audits)==len(cfg['jobs']) else 'first_batch_terminal_with_retained_failures',
                 active_processes=[],queued_jobs=[],completed_utc=datetime.now(timezone.utc).isoformat(),
                 result_directory=str(target.relative_to(root)),remaining_methods=['IncDE','SAGE','KGEditor','Rank','QP TRUE','QP TRUE+Prediction'])
    save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
