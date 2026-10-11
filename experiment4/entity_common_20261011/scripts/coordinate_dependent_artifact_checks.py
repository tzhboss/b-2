"""Wait without GPU claims, then delegate to the registered artifact queue."""
import argparse,hashlib,json,os,time
from pathlib import Path
import yaml
from coordinate_artifact_checks_v2 import main


def coordinate(root,config):
    cfg=json.loads((root/config).read_text());path=root/'experiments'/cfg['id']/'experiment.yaml'
    state=yaml.safe_load(path.read_text());assert state['status']=='registered'
    for name,value in state['frozen_input_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value
    state.update(status='waiting_registered_dependencies',coordinator_pid=os.getpid(),active_processes=[],jobs={})
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(yaml.safe_dump(state,sort_keys=False));temporary.replace(path)
    save()
    while True:
        ready=True
        for dependency in cfg['global_dependencies']:
            source=yaml.safe_load((root/dependency['canonical']).read_text())
            ready=ready and source['status'] in dependency['terminal_statuses'] and not source.get('active_processes')
        if ready:break
        time.sleep(3)
    # Same live coordinator continues into the queue; no worker held a GPU
    # during the dependency wait and no upstream process is interrupted.
    state['status']='registered';save();main(root,config)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();coordinate(a.root.resolve(),a.config)
