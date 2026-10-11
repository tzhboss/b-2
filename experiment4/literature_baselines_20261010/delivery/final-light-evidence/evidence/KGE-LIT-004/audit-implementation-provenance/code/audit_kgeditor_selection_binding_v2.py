#!/usr/bin/env python3
"""Additional read-only checkpoint epoch-to-monitor binding; does not replace the frozen CPU auditor."""
import argparse,hashlib,json,math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
PIN='ee004305ad0f01d985622833ba89a1fecaa339d752c3f1a4451af6d77b792b37'
MASTER='314b14fa671d140b579ba5e728afd6efb36349e34404d2f04a8ef33d2442a58b'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def assess(ep,step,score,per_epoch,monitors,epochs):
 failures=[]
 def ck(ok,msg):
  if not ok:failures.append(msg)
 ck(isinstance(ep,int) and 0<=ep<20,'selected epoch0..19')
 ck(step in [(ep+1)*per_epoch-1,(ep+1)*per_epoch],'selected checkpoint optimizer step corresponds to epoch')
 matched=[(s,v) for s,v in monitors if s in [step,step-1] and s in [(ep+1)*per_epoch-1,(ep+1)*per_epoch]]
 ck(bool(matched),'selected epoch has logged monitor at its checkpoint step')
 ck(bool(matched) and any(math.isclose(v,score,rel_tol=0,abs_tol=1e-7) for s,v in matched),'selected epoch monitor equals declared selection score')
 ck(bool(monitors) and math.isclose(max(v for s,v in monitors),score,rel_tol=0,abs_tol=1e-7),'declared score is global logged monitor maximum')
 expected=[(e+1)*per_epoch-1 for e in range(20)]
 ck(all(any(s in [x,x+1] for s,v in monitors) for x in expected),'one logged validation monitor for each completed epoch')
 ck(any(s in [step,step-1] and v==ep for s,v in epochs),'logged epoch tag links selected step to selected epoch')
 return failures,matched
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-audit',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists()
 assert sha(a.root/'code/audit_kgeditor_native_v2.py')==PIN
 manifest=a.root/'configs/experiments/KGE-LIT-004-jobs.json';assert sha(manifest)==MASTER
 report=json.loads(a.source_audit.read_text());assert report['status']=='PASS_WITH_LIMITATIONS'
 r=next(x for x in report['runs'] if x['run']==a.run_id);assert r['status']=='PASS_WITH_LIMITATIONS' and r['failure_count']==0 and not r['missing_evidence']
 run=a.root/'runs'/a.run_id;ev=r['evidence'];cp=ev['training_checkpoint'];path=Path(cp['best_model_path']);assert sha(path)==cp['checkpoint_sha256']
 complete=json.loads((run/'training-complete.json').read_text());assert complete['best_model_path']==str(path) and complete['best_model_score']==cp['best_model_score']
 monitors=[];epochs=[];bound={str(a.source_audit):sha(a.source_audit),str(path):sha(path),str(run/'training-complete.json'):sha(run/'training-complete.json')}
 for event in sorted((run/'logger').glob('**/events.out.tfevents.*')):
  bound[str(event)]=sha(event);acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload();tags=acc.Tags()['scalars']
  for tag,dest in [('Eval/hits3',monitors),('epoch',epochs)]:
   if tag in tags:dest.extend((x.step,x.value) for x in acc.Scalars(tag))
 per_epoch=1543 if 'fb15k237' in a.run_id else 745
 failures,matched=assess(cp['checkpoint_training_epoch'],cp['checkpoint_global_step'],cp['best_model_score'],per_epoch,monitors,epochs)
 unchanged=all(sha(k)==v for k,v in bound.items())
 if not unchanged:failures.append('bound evidence changed during audit')
 result=dict(status='PASS_SELECTION_BINDING' if not failures else 'FAIL',run=a.run_id,auditor_sha256=sha(__file__),source_auditor_sha256=PIN,source_audit_sha256=sha(a.source_audit),checks=8,failures=failures,selected_checkpoint_epoch=cp['checkpoint_epoch'],selected_checkpoint_step=cp['checkpoint_global_step'],selected_score=cp['best_model_score'],selected_monitor_events=matched,monitor_events=monitors,epoch_events=epochs,bound_input_sha256=bound,limitations=['Checks epoch-step-score association in logged evidence; no independent validation model inference or optimizer reconstruction.'],read_only=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(bool(failures))
