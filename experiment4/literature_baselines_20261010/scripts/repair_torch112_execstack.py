from pathlib import Path
import struct,shutil,hashlib,json
R=Path(__file__).resolve().parents[1];root=R/'environment/kgeditor38/lib/python3.8/site-packages/torch/lib';records=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
for lib in root.glob('*.so'):
 with lib.open('rb') as f:
  head=f.read(64)
  if head[:4]!=b'\x7fELF':continue
  assert head[4]==2 and head[5]==1
  off=struct.unpack_from('<Q',head,32)[0];size,num=struct.unpack_from('<HH',head,54)
  for i in range(num):
   f.seek(off+i*size);seg=f.read(size);typ,flags=struct.unpack_from('<II',seg,0)
   if typ==0x6474e551 and flags&1:
    assert lib.resolve().is_relative_to(R/'environment/kgeditor38');backup=R/'environment/torch112_original_elf'/lib.name;backup.parent.mkdir(exist_ok=True);assert not backup.exists();shutil.copyfile(lib,backup);before=sha(backup)
    with lib.open('r+b') as writer:writer.seek(off+i*size+4);writer.write(struct.pack('<I',flags&~1))
    records.append({'file':str(lib.relative_to(R)),'backup':str(backup.relative_to(R)),'original_sha256':before,'patched_sha256':sha(lib),'flag_offset':off+i*size+4,'old_flags':flags,'new_flags':flags&~1,'scope':'GNU_STACK ELFloader compatibility only, no tensor/code changes'})
(R/'environment/torch112-execstack-repair.json').write_text(json.dumps(records,indent=2));print(json.dumps(records),flush=True)
