"""Fetch pinned public assets; bind LFS SHA256 or Git blob OID, retain failures."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
root=a.root.resolve();cfg=json.loads((root/a.config).read_text());can=root/'experiments'/cfg['id']/'experiment.yaml'
s=yaml.safe_load(can.read_text());assert s['status']=='registered_asset_download'
for name,digest in s['frozen_input_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
s.update(status='running_asset_download',active_processes=[dict(pid=os.getpid(),stage='download original BERT pretraining assets',gpu=None)])
can.write_text(yaml.safe_dump(s,sort_keys=False));dest=root/cfg['destination'];dest.mkdir(parents=True,exist_ok=False)
def download(item):
    name=item['rfilename'];url=cfg['base_url']+'/'+name;path=dest/name
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix(path.suffix+'.partial')
    request=urllib.request.Request(url,headers={'User-Agent':'research-reproduction/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response,partial.open('xb') as out:
        while True:
            block=response.read(1024*1024)
            if not block:break
            out.write(block)
    assert partial.stat().st_size==item['size'],'File size '+name
    sha256=hashlib.sha256();blob=hashlib.sha1();blob.update(f'blob {item["size"]}\0'.encode())
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha256.update(block);blob.update(block)
    if item.get('lfs'):assert sha256.hexdigest()==item['lfs']['sha256'],'LFS SHA256 '+name
    else:assert blob.hexdigest()==item['blobId'],'Git blob OID '+name
    partial.rename(path);print('ASSET_VERIFIED',name,item['size'],sha256.hexdigest(),flush=True)
    return dict(name=name,size=item['size'],sha256=sha256.hexdigest(),upstream_git_blob=item['blobId'],upstream_lfs=item.get('lfs'))
try:
    with ThreadPoolExecutor(max_workers=3) as pool:verified=list(pool.map(download,cfg['files']))
    receipt=dict(repo=cfg['repo'],revision=cfg['revision'],files=verified,scope='Original language-model pretraining weights only; no KGE graph-finetuned checkpoint')
    (dest/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
except Exception:
    s.update(status='asset_download_failed_retained',active_processes=[]);can.write_text(yaml.safe_dump(s,sort_keys=False));raise
s.update(status='completed_verified_pretraining_assets_only',active_processes=[],download_receipt=str((dest/'download-receipt.json').relative_to(root)))
can.write_text(yaml.safe_dump(s,sort_keys=False))
