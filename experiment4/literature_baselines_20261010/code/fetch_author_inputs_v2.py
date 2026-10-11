import json,time,traceback,hashlib
from pathlib import Path
import gdown
ROOT=Path(__file__).resolve().parents[1]
receipts=[]
for label,url in [('datasets','https://drive.google.com/drive/folders/1K1gag6eTCJ-x7WM-zbn6IRGX_B9fy4hD'),('checkpoints','https://drive.google.com/drive/folders/1EOHdg8rC9iwgSyKl5RnEv9z6ATW5Ntbr')]:
 out=ROOT/'inputs'/'KGEditor'/label
 out.mkdir(parents=True,exist_ok=True)
 start=time.monotonic()
 try:
  files=gdown.download_folder(url=url,output=str(out)+'/',quiet=False)
  rec={'label':label,'url':url,'status':'DOWNLOADED' if files else 'NO_FILES','files':files,'elapsed_s':time.monotonic()-start}
 except BaseException:
  rec={'label':label,'url':url,'status':'FAIL','traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-start}
 receipts.append(rec);(ROOT/'environment'/'KGEditor-author-downloads-v2.json').write_text(json.dumps(receipts,indent=2))
 print('AUTHOR_INPUT_DOWNLOAD',label,rec['status'],flush=True)
