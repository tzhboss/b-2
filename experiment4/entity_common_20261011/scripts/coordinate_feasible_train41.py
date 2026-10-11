"""One-shot six-GPU experiment coordinator; no automatic restart of failures."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--launch',action='store_true');args=p.parse_args()
cfg=json.loads((root/args.config).read_text());canonical=root/'experiments'/cfg['id']/'experiment.yaml'
state=yaml.safe_load(canonical.read_text());assert state['status']=='registered'
if args.launch:
    launch=root/cfg['log_directory']/'launch.json';assert not launch.exists()
    log=(root/cfg['log_directory']/'coordinate.log').open('xb')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.Popen([os.path.abspath(root/cfg['python']),'-u',str(Path(__file__)),'--config',args.config],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    identity=dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]);launch.write_text(json.dumps(identity,indent=2));print(json.dumps(identity),flush=True)
    raise SystemExit(0)
import hashlib
for name,digest in state['frozen_input_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
idle={int(i) for line in usage.splitlines() for i,m,u in [line.split(',')] if int(m)<=1024 and int(u)<=20}
assert set(cfg['gpu_indices'])<=idle
state.update(status=cfg['running_status'],coordinator_pid=os.getpid(),jobs={});active={}
def save():
    state['active_processes']=[dict(id=k,pid=v['proc'].pid,start_ticks=v['ticks'],stage=v['stage'],gpu=v['gpu']) for k,v in active.items()]
    tmp=canonical.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(canonical)
def start(job,stage,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if stage in cfg['cpu_stages'] else str(gpu),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    log=(root/cfg['log_directory']/f"{job['id']}-{stage}.log").open('xb')
    proc=subprocess.Popen([os.path.abspath(root/cfg['python']),'-u',str(root/cfg['worker']),'--root',str(root),'--config',args.config,'--stage',stage,'--job',job['id']],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[job['id']]=dict(proc=proc,log=log,stage=stage,gpu=gpu,job=job,ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21])
    state['jobs'][job['id']]=dict(status='running',stage=stage,pid=proc.pid);print('START',job['id'],stage,proc.pid,gpu,flush=True)
for job,gpu in zip(cfg['jobs'],cfg['gpu_indices']):start(job,cfg['stages'][0],gpu)
save()
while active:
    for name,row in list(active.items()):
        rc=row['proc'].poll()
        if rc is None:continue
        row['log'].close();del active[name]
        index=cfg['stages'].index(row['stage'])
        if rc==0 and index+1<len(cfg['stages']):start(row['job'],cfg['stages'][index+1],row['gpu'])
        else:state['jobs'][name]=dict(status='completed_audited' if rc==0 else 'failed_retained',stage=row['stage'],exit_code=rc)
        print('TERMINAL_STAGE',name,row['stage'],rc,flush=True)
    save()
    if active:time.sleep(3)
dest=root/'results'/cfg['id'];dest.mkdir(exist_ok=False);rows=[]
for job in cfg['jobs']:
    if state['jobs'][job['id']]['status']=='completed_audited':
        out=root/cfg['output_directory']/job['id']
        record=dict(job=job,receipt=json.loads((out/'receipt.json').read_text()),audit=json.loads((out/'audit.json').read_text()))
        for name in cfg.get('extra_results',[]):record[name]=json.loads((out/(name+'.json')).read_text())
        rows.append(record)
(dest/'summary.json').write_text(json.dumps(dict(rows=rows,completed_audited_jobs=len(rows),planned_jobs=len(cfg['jobs']),scientific_utility_asserted=False),indent=2))
state.update(status=cfg['terminal_status'] if len(rows)==len(cfg['jobs']) else 'terminal_with_retained_failures',result_directory=str(dest.relative_to(root)),completed_utc=datetime.now(timezone.utc).isoformat());save()
if len(rows)==len(cfg['jobs']):
    aggregate=subprocess.run([os.path.abspath(root/cfg['python']),str(root/'scripts/report_feasible_train41.py')],cwd=root)
    state['aggregation_exit_code']=aggregate.returncode;save()
