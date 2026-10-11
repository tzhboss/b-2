import argparse,hashlib,json,os,subprocess,time,fcntl,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
CANON=ROOT/'experiments/KGE-LIT-001/experiment.yaml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_state(state):
    tmp=CANON.with_suffix('.yaml.runtime.tmp')
    tmp.write_text(yaml.safe_dump(state,sort_keys=False,allow_unicode=True));tmp.replace(CANON)
def verified(job):
    for name,digest in job['input_sha256'].items():assert sha(ROOT/name)==digest,name
    assert Path(job['cwd']).resolve().is_relative_to(ROOT)
    assert (ROOT/job['run_directory']).resolve().is_relative_to(ROOT)
def gpu_info():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if r.returncode:return {}
    return {int(a):{'free_mb':int(b),'utilization':int(c)} for a,b,c in (row.split(',') for row in r.stdout.splitlines())}
def main(jobfile):
    lock=(ROOT/('environment/scheduler-'+CANON.parent.name+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=yaml.safe_load(CANON.read_text());assert state['registration_status']=='REGISTERED'
    assert sha(jobfile)==state['jobs_manifest_sha256']
    assert sha(__file__)==state['scheduler_sha256']
    jobs=json.loads(Path(jobfile).read_text());assert len({j['id'] for j in jobs})==len(jobs)
    assert state['active_processes']==[]
    states={j['id']:{'id':j['id'],'method':j['method'],'status':'pending','gpu':j['gpu']} for j in jobs}
    state['jobs']=list(states.values());state['status']='running';state['scheduler_pid']=os.getpid();write_state(state)
    active={};remaining=list(jobs)
    try:
        while remaining or active:
            free=gpu_info();busy={job['gpu'] for job,p,fh in active.values()}
            for job in list(remaining):
                gid=job['gpu']
                if gid in busy or gid not in free or free[gid]['free_mb']<job.get('minimum_free_gpu_mb',4000):continue
                verified(job);out=ROOT/job['run_directory'];assert not out.exists();out.mkdir(parents=True)
                (out/'registered-job.json').write_text(json.dumps(job,indent=2))
                for directory in job.get('create_directories',[]):
                    target=out/directory;assert target.resolve().is_relative_to(out);target.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':str(gid),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
                for k,v in job.get('environment',{}).items():env[k]=str(v)
                runtime_cwd=Path(job['cwd']);runtime_command=list(job['command'])
                if job.get('copy_repository'):
                    source=(ROOT/job['copy_repository']).resolve();assert source.is_relative_to(ROOT/'repos')
                    assert runtime_cwd.is_relative_to(source)
                    snapshot=out/'worktree';shutil.copytree(source,snapshot,ignore=shutil.ignore_patterns('.git','__pycache__'))
                    runtime_cwd=snapshot/runtime_cwd.relative_to(source)
                    runtime_command=[str(snapshot)+arg[len(str(source)):] if arg.startswith(str(source)+'/') else arg for arg in runtime_command]
                    pairs={}
                    for name,digest in job['input_sha256'].items():
                        original=(ROOT/name).resolve()
                        if original.is_relative_to(source):
                            copied=snapshot/original.relative_to(source);assert sha(copied)==digest
                            pairs[str(copied.relative_to(out))]=digest
                    (out/'copied-source-sha256.json').write_text(json.dumps(pairs,indent=2))
                (out/'runtime-command.json').write_text(json.dumps({'command':runtime_command,'cwd':str(runtime_cwd),'gpu':gid},indent=2))
                fh=(out/'run.log').open('xb');proc=subprocess.Popen(runtime_command,cwd=runtime_cwd,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
                rec=states[job['id']];rec.update(status='running',pid=proc.pid,started_monotonic=time.monotonic(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),log=str((out/'run.log').relative_to(ROOT)))
                active[job['id']]=(job,proc,fh);busy.add(gid);remaining.remove(job);print('JOB_STARTED',job['id'],gid,proc.pid,flush=True)
            for jid,(job,proc,fh) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                fh.close();rec=states[jid];rec.update(status='terminal_unreviewed' if code==0 else 'failed',exit_code=code,elapsed_s=time.monotonic()-rec['started_monotonic'],terminal_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                try:verified(job);rec['source_inputs_unchanged']=True
                except BaseException as err:rec.update(status='failed',source_inputs_unchanged=False,source_error=repr(err))
                out=ROOT/job['run_directory'];rec['log_sha256']=sha(out/'run.log');(out/'terminal.json').write_text(json.dumps(rec,indent=2));del active[jid]
                print('JOB_TERMINAL',jid,code,round(rec['elapsed_s'],2),flush=True)
            state['active_processes']=[{'id':jid,'pid':p.pid,'gpu':j['gpu'],'stage':'native_training_evaluation'} for jid,(j,p,f) in active.items()]
            state['resource_observation']={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpus':free}
            write_state(state)
            if remaining or active:time.sleep(15)
        state.update(status='producer_terminal_pending_audit',active_processes=[]);write_state(state);print('ALL_REGISTERED_JOBS_TERMINAL',flush=True)
    except BaseException as err:
        state.update(status='controller_failed',controller_error=repr(err));write_state(state);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('jobs',type=Path);p.add_argument('--canonical',type=Path,default=CANON);a=p.parse_args();CANON=a.canonical.resolve();assert CANON.is_relative_to(ROOT/'experiments');main(a.jobs.resolve())
