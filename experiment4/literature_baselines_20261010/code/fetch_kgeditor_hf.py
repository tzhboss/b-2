import pathlib,json,urllib.request,concurrent.futures,hashlib,time
R=pathlib.Path(__file__).resolve().parents[1]
tasks=[]
for typ,repo,rev in [('models','zjunlp/KGEditor','a103204def5465ac090935c925451eef6692f2cc'),('datasets','zjunlp/KGEditor','18a560856826fdcfab5b328829f550ce0f9e5a7a')]:
 for f in json.loads((R/'environment'/(typ+'-KGEditor-tree.json')).read_text()):
  if f['type']!='file' or not (f['path'].startswith('E-') or f['path']=='README.md'):continue
  tasks.append((typ,repo,rev,f))
def get(task):
 typ,repo,rev,f=task;url='https://huggingface.co/'+('datasets/' if typ=='datasets' else '')+repo+'/resolve/'+rev+'/'+f['path']
 target=R/('data/KGEditor-HF' if typ=='datasets' else 'models/KGEditor-HF')/f['path'];target.parent.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();rec={'type':typ,'repo':repo,'revision':rev,'path':f['path'],'url':url}
 try:
  with urllib.request.urlopen(url,timeout=60) as response,target.with_suffix(target.suffix+'.partial').open('wb') as out:
   while chunk:=response.read(4*1024*1024):out.write(chunk)
  temp=target.with_suffix(target.suffix+'.partial');assert temp.stat().st_size==f['size']
  h=hashlib.sha256();g=hashlib.sha1();g.update(('blob '+str(f['size'])+'\0').encode())
  with temp.open('rb') as stream:
   while chunk:=stream.read(4*1024*1024):h.update(chunk);g.update(chunk)
  assert h.hexdigest()==f['lfs']['oid'] if 'lfs' in f else g.hexdigest()==f['oid']
  temp.replace(target);rec.update(status='PASS',sha256=h.hexdigest(),bytes=f['size'])
 except Exception as e:rec.update(status='FAIL',error=repr(e))
 rec['elapsed_s']=time.monotonic()-start;print(rec['status'],typ,f['path'],flush=True);return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(get,tasks))
(R/'environment/KGEditor-HF-download-manifest.json').write_text(json.dumps(records,indent=2))
