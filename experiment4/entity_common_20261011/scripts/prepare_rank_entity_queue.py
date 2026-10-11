"""Prepare old TRAIN guards, then dispatch the dependent registered queue."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import yaml
from hashlib import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();root=a.root.resolve()
    cfg=json.loads((root/a.config).read_text());can=root/'experiments'/cfg['id']/'experiment.yaml';s=yaml.safe_load(can.read_text());assert s['status']=='registered_feature_preparation'
    for name,digest in s['frozen_input_sha256'].items():assert sha256((root/name).read_bytes()).hexdigest()==digest
    for name,digest in s['external_source_sha256'].items():assert sha256(Path(name).read_bytes()).hexdigest()==digest
    def save():
        temp=can.with_suffix('.tmp');temp.write_text(yaml.safe_dump(s,sort_keys=False));temp.replace(can)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    python=Path(os.path.abspath(root/cfg['python']));s.update(status='running_feature_preparation',active_processes=[dict(pid=os.getpid(),gpu=a.gpu,stage='TRAIN-only guards')]);save()
    log=(root/cfg['log_directory']/'feature-controls.log').open('xb')
    control=subprocess.run([str(python),str(root/'scripts/check_rank_entity_controls.py'),'--root',str(root),'--config',a.config],env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
    if control.returncode:s.update(status='failed_controls_retained',active_processes=[]);save();sys.exit(control.returncode)
    for seed in sorted({j['seed'] for j in cfg['jobs']}):
        log=(root/cfg['log_directory']/f'features-s{seed}.log').open('xb')
        cmd=[str(python),'-u',str(root/'scripts/run_rank_entity.py'),'--root',str(root),'--config',a.config,'--stage','features','--seed',str(seed)]
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        s.update(active_processes=[dict(pid=proc.pid,gpu=a.gpu,stage='TRAIN-only guards',seed=seed)]);save();rc=proc.wait();log.close()
        if rc:s.update(status='failed_features_retained',active_processes=[]);save();sys.exit(rc)
    s.update(status='registered',active_processes=[],feature_preparation_complete=True);save()
    # No next stage runs unless all three TRAIN guard sets and controls pass.
    subprocess.run([str(python),'-u',str(root/'scripts/coordinate_gpu_claim_queue.py'),'--root',str(root),'--config',a.config],env=env,check=True)


if __name__=='__main__':main()
