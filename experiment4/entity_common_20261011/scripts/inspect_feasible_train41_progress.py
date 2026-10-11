"""Read canonical plus live Linux process identity; never restart workers."""
import json,re,subprocess
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
state=yaml.safe_load((root/'experiments/KGE-ENTITY-COMMON-FEASIBLE-TRAIN-041/experiment.yaml').read_text())
rows=[]
for active in state['active_processes']:
    proc=Path('/proc')/str(active['pid']);alive=proc.exists();identity=False;command=None
    if alive:
        # All launched executable paths here contain no spaces or parentheses.
        stat=(proc/'stat').read_text();ticks=stat.rsplit(')',1)[1].split()[19]
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        identity=(ticks==str(active['start_ticks']) and active['id'] in command and 'run_feasible_train41.py' in command)
    namespace_pid=None
    if alive:
        for line in (proc/'status').read_text().splitlines():
            if line.startswith('NSpid:'):namespace_pid=int(line.split()[-1])
    # Training progress remains in the run log after evaluation/audit starts.
    log=root/'logs/feasible-train41'/f"{active['id']}-run.log"
    lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    progress=[line for line in lines if line.startswith('FEASIBLE_TRAIN_PROGRESS ')]
    completed=int(progress[-1].split()[2]) if progress else 0
    accepted=sum(line.split()[4]=='True' for line in progress)
    rows.append(dict(id=active['id'],stage=active['stage'],pid=active['pid'],namespace_pid=namespace_pid,assigned_gpu=active['gpu'],process_identity_matches=identity,
           updates=completed,total_updates=3753,accepted_so_far=accepted,last_progress=progress[-1] if progress else None))
launch=json.loads((root/'logs/feasible-train41/launch.json').read_text());proc=Path('/proc')/str(launch['pid'])
coordinator_live=proc.exists() and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==launch['start_ticks']
gpu_processes=[]
output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
for line in output.splitlines():
    fields=[s.strip() for s in line.split(',')]
    if len(fields)==3 and fields[0].isdigit():gpu_processes.append(dict(host_pid=int(fields[0]),gpu_uuid=fields[1],memory=fields[2]))
gpu_ids={}
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
for line in inventory.splitlines():
    fields=[s.strip() for s in line.split(',')]
    if len(fields)==2 and fields[0].isdigit():gpu_ids[int(fields[0])]=fields[1]
for row in rows:
    row['assigned_gpu_uuid']=gpu_ids.get(row['assigned_gpu'])
    row['driver_compute_rows_on_assigned_gpu']=[p for p in gpu_processes if p['gpu_uuid']==row['assigned_gpu_uuid']]
    row['driver_pid_namespace_mapping_verified']=False
print(json.dumps(dict(status=state['status'],coordinator_identity_matches=coordinator_live,active=rows,
    finished_jobs={k:v for k,v in state['jobs'].items() if v['status']!='running'}),indent=2))
