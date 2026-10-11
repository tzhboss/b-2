"""Launch native author training with config-relative paths; no reduced epochs."""
import argparse, hashlib, json, os, pathlib, shlex, subprocess, sys, time
p=argparse.ArgumentParser()
p.add_argument('--config', default=str(pathlib.Path(__file__).with_name('config.json')))
p.add_argument('--gpu', type=int, required=True, help='physical GPU assigned by controller')
p.add_argument('--run-name', required=True)
p.add_argument('--dry-run', action='store_true')
a=p.parse_args()
config_path=pathlib.Path(a.config).resolve(); base=config_path.parent
c=json.loads(config_path.read_text()); repo=(base/c['repo_root']).resolve()
assert a.gpu >= 0
assert pathlib.Path(a.run_name).name == a.run_name and a.run_name not in {'.','..'}
run=(base/c['output_root']/a.run_name).resolve()
cmd=[sys.executable,'-u',*c['command']]
for key in ['save_path','log_path']:
    folder=run/('checkpoints' if key=='save_path' else 'logs')
    cmd.extend(['-'+key,os.path.relpath(folder,repo)+'/'])
cmd.extend(['-data_path',os.path.relpath(repo/'data',repo)+'/'])
cmd.extend(['-observation_path',os.path.relpath(run/'observations',repo)])
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
env['PYTHONPATH']=str((base/c['vendor']).resolve())+os.pathsep+env.get('PYTHONPATH','')
env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONHASHSEED']='3407'; env['OMP_NUM_THREADS']=str(c['threads']); env['MKL_NUM_THREADS']=str(c['threads']); env['OPENBLAS_NUM_THREADS']=str(c['threads'])
print('cwd:',repo,flush=True); print('CUDA_VISIBLE_DEVICES:',a.gpu,flush=True); print(shlex.join(cmd),flush=True)
if a.dry_run: sys.exit(0)
actual_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert actual_commit==c['upstream_commit'], (actual_commit,c['upstream_commit'])
run.mkdir(parents=True,exist_ok=False)
(run/'checkpoints').mkdir(); (run/'logs').mkdir()
manifest={'command':cmd,'cwd':str(repo),'assigned_gpu':a.gpu,'config':c,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'upstream_commit':actual_commit,'note':'execution evidence only; canonical and promotion owned by controller'}
(run/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.monotonic()
with (run/'stdout.log').open('w',buffering=1) as out:
    result=subprocess.run(cmd,cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT)
(run/'process_exit.json').write_text(json.dumps({'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
sys.exit(result.returncode)
