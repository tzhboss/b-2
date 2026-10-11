"""Launch once; authoritative state/process identity prevents duplicate work."""
import json,os,subprocess
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'configs/feasible-train40.json').read_text())
state=yaml.safe_load((root/'experiments'/cfg['id']/'experiment.yaml').read_text())
assert state['status']=='registered' and not state['active_processes']
launch=root/cfg['log_directory']/'launch.json';assert not launch.exists()
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
log=(root/cfg['log_directory']/'coordinate.log').open('xb')
proc=subprocess.Popen([os.path.abspath(root/cfg['python']),'-u',str(root/'scripts/run_feasible_train40.py'),'--root',str(root),'--stage','coordinate'],
    cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
identity=dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21],command='run_feasible_train40.py --stage coordinate')
launch.write_text(json.dumps(identity,indent=2));print(json.dumps(identity),flush=True)
