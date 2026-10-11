from pathlib import Path
import os,sys,json,time,fcntl,subprocess,hashlib,datetime
import yaml
R=Path(__file__).resolve().parents[1]
C=R/'experiments/KGE-LIT-004-AUDIT/experiment.yaml'
NATIVE=R/'experiments/KGE-LIT-004/experiment.yaml'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4194304):h.update(b)
 return h.hexdigest()
def save(c):
 p=C.with_suffix('.yaml.tmp');p.write_text(yaml.safe_dump(c,sort_keys=False));p.replace(C)
def inputs_ok(c):
 for n,d in c['implementation_sha256'].items():assert sha(R/n)==d,n
 assert sha(R/'configs/experiments/KGE-LIT-004-jobs.json')==c['native_master_sha256']
def gpu_free(gpu):
 p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if p.returncode!=0:return False
 for line in p.stdout.splitlines():
  try:i,m,u=map(int,map(str.strip,line.split(',')))
  except ValueError:continue
  if i==gpu:return m>=23000 and u<=20
 return False
lock=(R/'environment/scheduler-KGE-LIT-004-AUDIT.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=yaml.safe_load(C.read_text());inputs_ok(c);assert sha(__file__)==c['coordinator_sha256'];jobs=json.loads((R/'configs/experiments/KGE-LIT-004-jobs.json').read_text());records={j['id']:{'id':j['id'],'gpu':j['gpu'],'stage':'waiting_for_native_terminal'} for j in jobs};pending=list(jobs);active={};ready=[];c['coordinator_pid']=os.getpid()
try:
 while pending or active or ready:
  native=yaml.safe_load(NATIVE.read_text());busy={j['gpu'] for j in native.get('active_processes',[])}
  for j in list(pending):
   run=R/j['run_directory'];t=run/'terminal.json'
   if not t.exists() or j['gpu'] in busy:continue
   terminal=json.loads(t.read_text());assert terminal['id']==j['id'];pending.remove(j);rec=records[j['id']]
   if terminal['exit_code']!=0 or not terminal.get('source_inputs_unchanged'):
    rec.update(stage='native_failed',status='FAIL',producer_exit_code=terminal['exit_code']);continue
   inputs_ok(c);env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   dest=R/'logs'/(j['id']+'-independent-cpu-audit.json');err=R/'logs'/(j['id']+'-independent-cpu-audit.stderr.log');out=dest.open('xb');stderr=err.open('xb');proc=subprocess.Popen([str(R/'environment/kgeditor38/bin/python'),'-u',str(R/'code/audit_kgeditor_native.py'),'--root',str(R),'--runs',j['id']],cwd=R,env=env,stdout=out,stderr=stderr,start_new_session=True);active[j['id']]={'job':j,'stage':'cpu_audit','proc':proc,'out':out,'err':stderr,'report':dest,'started':time.monotonic()};rec.update(stage='independent_cpu_audit',pid=proc.pid);print('CPU_AUDIT_STARTED',j['id'],proc.pid,flush=True)
  for jid,a in list(active.items()):
   code=a['proc'].poll()
   if code is None:continue
   a['out'].close();a['err'].close();rec=records[jid];rec[a['stage']+'_exit_code']=code;rec[a['stage']+'_elapsed_s']=time.monotonic()-a['started'];rec[a['stage']+'_report']=str(a['report'].relative_to(R));del active[jid]
   if code!=0 or not a['report'].exists():rec.update(stage=a['stage']+'_failed',status='FAIL');continue
   report=json.loads(a['report'].read_text())
   expected='PASS_WITH_LIMITATIONS' if a['stage']=='cpu_audit' else 'PASS_SAMPLE_REPLAY_WITH_LIMITATIONS'
   if report['status']!=expected:rec.update(stage=a['stage']+'_failed',status='FAIL');continue
   rec[a['stage']+'_report_sha256']=sha(a['report'])
   if a['stage']=='cpu_audit':
    assert len(report['runs'])==1 and report['runs'][0]['run']==jid and report['runs'][0]['failure_count']==0
    ready.append((a['job'],a['report']));rec.update(stage='waiting_for_free_gpu_replay');print('CPU_AUDIT_PASS',jid,flush=True)
   else:
    rec.update(stage='completed_sample_audited',status='PASS_WITH_LIMITATIONS',checked_examples=report['checked_examples'],max_absolute_logit_error=report['max_absolute_logit_error']);print('SAMPLE_REPLAY_PASS',jid,report['checked_examples'],flush=True)
  for j,audit in list(ready):
   if j['gpu'] in busy or not gpu_free(j['gpu']):continue
   inputs_ok(c);jid=j['id'];work=R/'audit-replays'/(jid+'-v1');assert not work.exists();env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(j['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   out=(R/'logs'/(jid+'-sample-replay.stdout.log')).open('xb');stderr=(R/'logs'/(jid+'-sample-replay.stderr.log')).open('xb');proc=subprocess.Popen([str(R/'environment/kgeditor38/bin/python'),'-u',str(R/'code/replay_kgeditor_samples.py'),'--root',str(R),'--run',str(R/j['run_directory']),'--source-audit',str(audit),'--work-dir',str(work),'--device','cuda:0'],cwd=R,env=env,stdout=out,stderr=stderr,start_new_session=True);active[jid]={'job':j,'stage':'sample_replay','proc':proc,'out':out,'err':stderr,'report':work/'replay-report.json','started':time.monotonic()};records[jid].update(stage='independent_sample_replay',pid=proc.pid);ready.remove((j,audit));print('SAMPLE_REPLAY_STARTED',jid,proc.pid,flush=True)
  c.update(status='running_audits' if active else 'waiting_for_inputs_or_free_gpu',jobs=list(records.values()),active_processes=[{'id':jid,'pid':a['proc'].pid,'gpu':None if a['stage']=='cpu_audit' else a['job']['gpu'],'stage':a['stage']} for jid,a in active.items()]);save(c)
  if pending or active or ready:time.sleep(15)
 c.update(status='all_audits_terminal_pending_controller_review',active_processes=[]);save(c);print('ALL_AUDITS_TERMINAL',flush=True)
except BaseException as err:
 c.update(status='audit_controller_failed',controller_error=repr(err),jobs=list(records.values()));save(c);raise
