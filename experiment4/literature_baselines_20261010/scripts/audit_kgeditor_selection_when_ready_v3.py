from pathlib import Path
import json,hashlib,yaml,fcntl,os,time,subprocess
R=Path(__file__).resolve().parents[1];C=R/'experiments/KGE-LIT-004-SELECTION-V3/experiment.yaml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(c):
 p=C.with_suffix('.yaml.tmp');p.write_text(yaml.safe_dump(c,sort_keys=False));p.replace(C)
lock=(R/'environment/scheduler-KGE-LIT-004-SELECTION-V3.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=yaml.safe_load(C.read_text());assert sha(__file__)==c['coordinator_sha256'];assert sha(R/'code/audit_kgeditor_selection_binding_v2.py')==c['auditor_sha256'];c['coordinator_pid']=os.getpid();save(c)
while any(j['stage']=='waiting_for_cpu_audit' for j in c['jobs']):
 state=yaml.safe_load((R/'experiments/KGE-LIT-004-AUDIT-V3/experiment.yaml').read_text());records={j['id']:j for j in state['jobs']}
 for j in c['jobs']:
  if j['stage']!='waiting_for_cpu_audit':continue
  original=records[j['id']];stage=original['stage']
  if stage in ['cpu_audit_failed','native_failed'] or state['status']=='audit_controller_failed':
   j.update(stage='blocked_prerequisite_failed',status='NO_PASS',prerequisite=stage);save(c);continue
  if original.get('cpu_audit_exit_code')!=0 or not original.get('cpu_audit_report_sha256'):continue
  source=R/original['cpu_audit_report'];assert sha(source)==original['cpu_audit_report_sha256'];assert json.loads(source.read_text())['status']=='PASS_WITH_LIMITATIONS'
  dest=R/'logs'/(j['id']+'-selection-binding-v3.json');stdout=R/'logs'/(j['id']+'-selection-binding-v3.stdout.log');stderr=R/'logs'/(j['id']+'-selection-binding-v3.stderr.log');assert not dest.exists()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with stdout.open('xb') as out,stderr.open('xb') as err:
   proc=subprocess.Popen([str(R/'environment/kgeditor38/bin/python'),str(R/'code/audit_kgeditor_selection_binding_v2.py'),'--root',str(R),'--source-audit',str(source),'--run-id',j['id'],'--output',str(dest)],env=env,cwd=R,stdout=out,stderr=err,start_new_session=True)
   c['active_processes']=[dict(id=j['id'],pid=proc.pid,stage='selection_binding',gpu=None)];j.update(stage='running',pid=proc.pid);save(c);code=proc.wait()
  report=json.loads(dest.read_text()) if dest.exists() else {};j.update(stage='terminal',exit_code=code,status=report.get('status','FAIL_WITHOUT_REPORT'),report=str(dest.relative_to(R)),report_sha256=sha(dest) if dest.exists() else None);c['active_processes']=[];save(c);print('SELECTION_TERMINAL',j['id'],j['status'],flush=True)
 if any(j['stage']=='waiting_for_cpu_audit' for j in c['jobs']):time.sleep(15)
c['status']='terminal_pending_controller_review';save(c)
