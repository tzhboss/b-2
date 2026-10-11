"""Rebuild existing experiment4 audits into a new directory; no training."""
from pathlib import Path
import os, shutil, subprocess, hashlib, json
base=Path('/data/lc/tzh/experiment4')
dst=base/'comparison20261010'
assert not dst.exists(), 'Preserve existing comparison directory'
(dst/'code').mkdir(parents=True)
(dst/'results').mkdir()
for name in ('audit_rank_kernel_v2.py','audit_remote_integrity.py'):
    shutil.copy2(base/'code'/name,dst/'code'/name)
for name in ('data','checkpoints'):
    (dst/name).symlink_to(base/name,target_is_directory=True)
original=list((base/'results').glob('*_edit*.json'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in original}
for p in original:
    (dst/'results'/p.name).symlink_to(p)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
py='/data/conda/envs/test/bin/python'
for name,args in [('audit_remote_integrity.py',[]),('audit_rank_kernel_v2.py',['--work',str(dst)])]:
    subprocess.run([py,str(dst/'code'/name),*args],check=True,env=env)
assert before=={str(p):sha(p) for p in original}
compared=[]
for name in ('phase24_integrity.json','phase24_v2_rank_kernel_audit.json','phase24_v2_rank_kernel_audit.csv'):
    assert (dst/'results'/name).read_bytes()==(base/'results'/name).read_bytes(),name
    compared.append({'file':name,'sha256':sha(dst/'results'/name),'byte_identical':True})
manifest={'status':'PASS','scope':'Saved-result aggregation and hash replay; no new model inference or training',
          'original_edit_files_unchanged':len(original),'outputs':compared,
          'source_script_hashes':{p.name:sha(p) for p in (dst/'code').glob('*.py')}}
(dst/'replay_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps(manifest))
