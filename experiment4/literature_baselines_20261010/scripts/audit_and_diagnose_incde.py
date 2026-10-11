import pathlib,subprocess,time,json,hashlib,os,fcntl
import yaml
R=pathlib.Path(__file__).resolve().parents[1];CANON=R/'experiments/KGE-LIT-005/experiment.yaml';SOURCE_CANON=R/'experiments/KGE-LIT-001/experiment.yaml'
def sha(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def save(state):
 tmp=CANON.with_suffix('.yaml.coordinator.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(CANON)
lock=(R/'environment/scheduler-KGE-LIT-005.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=yaml.safe_load(CANON.read_text());assert sha(__file__)==state['coordinator_sha256'];assert sha(R/'code/audit_incde_native.py')==state['required_auditor_sha256'];assert sha(R/'code/evaluate_candidate_expansion.py')==state['implementation_sha256']
manifest=R/'configs/experiments/KGE-LIT-001-jobs.json';assert sha(manifest)=='7d211e0780af166f027623f57668b0ab5e61f160dce6cc875baca228b88a3951';jobs=json.loads(manifest.read_text());pending=list(jobs);active={};states={j['id']:{'id':j['id'],'gpu':j['gpu'],'stage':'waiting_for_native_terminal'} for j in jobs};state['coordinator_pid']=os.getpid();save(state)
while pending or active:
 source=yaml.safe_load(SOURCE_CANON.read_text());busy={j['gpu'] for j in source.get('active_processes',[])}
 for job in list(pending):
  run=R/job['run_directory'];term=run/'terminal.json'
  if not term.exists() or job['gpu'] in busy:continue
  receipt=json.loads(term.read_text());assert receipt['id']==job['id'];pending.remove(job);rec=states[job['id']]
  if receipt['exit_code']!=0:rec.update(stage='native_failed',status='FAIL');continue
  env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':str(job['gpu']),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
  existing=R/'results/KGE-LIT-001-audit'/job['id']/'audit.json'
  if existing.exists():
   audited=json.loads(existing.read_text());assert audited['status']=='PASS' and audited['id']==job['id'] and audited['audit_code_sha256']==state['required_auditor_sha256'] and audited['registration_sha256']==sha(manifest)
   assert not (R/'runs/candidate-diagnostic'/job['id']).exists()
   log=(R/'logs'/(job['id']+'-candidate-diagnostic-attempt2.log')).open('xb');proc=subprocess.Popen([str(R/'environment/downloads/bin/python'),'-u',str(R/'code/evaluate_candidate_expansion.py'),str(run)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec.update(stage='candidate_diagnostic',pid=proc.pid);active[job['id']]={'job':job,'proc':proc,'log':log,'env':env,'stage':'diagnostic'};print('DIAGNOSTIC_RETRY_STARTED',job['id'],proc.pid,flush=True);continue
  assert not existing.parent.exists()
  log=(R/'logs'/(job['id']+'-audit.log')).open('xb');proc=subprocess.Popen([str(R/'environment/downloads/bin/python'),'-u',str(R/'code/audit_incde_native.py'),str(run)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec.update(stage='independent_native_audit',pid=proc.pid);active[job['id']]={'job':job,'proc':proc,'log':log,'env':env,'stage':'audit'};print('AUDIT_STARTED',job['id'],proc.pid,flush=True)
 for jid,a in list(active.items()):
  code=a['proc'].poll()
  if code is None:continue
  a['log'].close();job=a['job'];rec=states[jid]
  if a['stage']=='audit':
   report=R/'results/KGE-LIT-001-audit'/jid/'audit.json'
   if code!=0 or not report.exists() or json.loads(report.read_text())['status']!='PASS':
    rec.update(stage='audit_failed',status='FAIL',exit_code=code);del active[jid];continue
   log=(R/'logs'/(jid+'-candidate-diagnostic.log')).open('xb');proc=subprocess.Popen([str(R/'environment/downloads/bin/python'),'-u',str(R/'code/evaluate_candidate_expansion.py'),str(R/job['run_directory'])],cwd=R,env=a['env'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a.update(proc=proc,log=log,stage='diagnostic');rec.update(stage='candidate_diagnostic',pid=proc.pid);print('DIAGNOSTIC_STARTED',jid,proc.pid,flush=True)
  else:
   report=R/'runs/candidate-diagnostic'/jid/'terminal.json';rec.update(stage='terminal_unreviewed' if code==0 and report.exists() else 'diagnostic_failed',status='TERMINAL_PENDING_AUDIT' if code==0 and report.exists() else 'FAIL',exit_code=code);del active[jid];print('DIAGNOSTIC_TERMINAL',jid,code,flush=True)
 state['status']='running' if active else 'waiting_for_native_inputs';state['jobs']=list(states.values());state['active_processes']=[{'id':jid,'pid':a['proc'].pid,'gpu':a['job']['gpu'],'stage':a['stage']} for jid,a in active.items()];save(state)
 if pending or active:time.sleep(15)
state.update(status='producer_terminal_pending_audit',active_processes=[]);save(state);print('ALL_DIAGNOSTICS_TERMINAL',flush=True)
