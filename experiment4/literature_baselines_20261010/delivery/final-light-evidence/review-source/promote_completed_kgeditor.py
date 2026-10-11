"""Controller-only terminal promotion. Fails before writing if any required evidence is incomplete."""
from pathlib import Path
import json,yaml,hashlib,fcntl,shutil,datetime,math
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def assert_manifest(path):
 for name,digest in load(path/'manifest-sha256.json').items():assert sha(path/name)==digest,str(path/name)
def close_mapping(a,b):
 assert set(a)==set(b)
 for k,v in a.items():
  if isinstance(v,float):assert math.isclose(v,b[k],rel_tol=0,abs_tol=1e-12),k
  else:assert v==b[k],k
locks=[];states={}
for eid in ['KGE-LIT-004','KGE-LIT-004-AUDIT-V3','KGE-LIT-004-SELECTION-V3']:
 lock=(R/('environment/scheduler-'+eid+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 c=yaml.safe_load((R/'experiments'/eid/'experiment.yaml').read_text());assert not c.get('active_processes'),eid;states[eid]=c
native=states['KGE-LIT-004'];audit=states['KGE-LIT-004-AUDIT-V3'];selection=states['KGE-LIT-004-SELECTION-V3']
assert native['status']=='producer_terminal_pending_audit',native['status']
assert audit['status']=='all_audits_terminal_pending_controller_review',audit['status']
assert selection['status']=='terminal_pending_controller_review',selection['status']
master=R/'configs/experiments/KGE-LIT-004-jobs.json';assert sha(master)==native['jobs_manifest_sha256']=='314b14fa671d140b579ba5e728afd6efb36349e34404d2f04a8ef33d2442a58b'
for relative,digest in audit['implementation_sha256'].items():assert sha(R/relative)==digest,relative
assert sha(R/'scripts/audit_and_replay_kgeditor_v3.py')==audit['coordinator_sha256']
assert sha(R/'scripts/audit_kgeditor_selection_when_ready_v3.py')==selection['coordinator_sha256']
assert sha(R/'code/audit_kgeditor_selection_binding_v2.py')==selection['auditor_sha256']
records={r['id']:r for r in audit['jobs']};selections={r['id']:r for r in selection['jobs']};native_records={r['id']:r for r in native['jobs']};jobs=load(master);prepared=[]
for job in jobs:
 jid=job['id'];run=R/job['run_directory'];rec=records[jid];sel=selections[jid];t=load(run/'terminal.json')
 assert t['status']=='terminal_unreviewed' and t['exit_code']==0 and t['source_inputs_unchanged'] and native_records[jid]['exit_code']==0
 assert native_records[jid]['status']=='terminal_unreviewed' and sha(run/'run.log')==t['log_sha256']
 assert rec['stage']=='completed_sample_audited' and rec['status']=='PASS_WITH_LIMITATIONS' and rec['cpu_audit_exit_code']==rec['sample_replay_exit_code']==0
 assert sel['stage']=='terminal' and sel['status']=='PASS_SELECTION_BINDING' and sel['exit_code']==0
 for rel,digest in job['input_sha256'].items():assert sha(R/rel)==digest,rel
 for rel,digest in load(run/'copied-source-sha256.json').items():assert sha(run/rel)==digest,rel
 for key in ['cpu_audit','sample_replay']:
  assert sha(R/rec[key+'_report'])==rec[key+'_report_sha256'],key
 assert sha(R/sel['report'])==sel['report_sha256']
 cpu=load(R/rec['cpu_audit_report']);rp=load(R/rec['sample_replay_report']);sp=load(R/sel['report'])
 assert cpu['status']=='PASS_WITH_LIMITATIONS' and cpu['master']['failure_count']==0 and len(cpu['runs'])==1
 cr=cpu['runs'][0];assert cr['run']==jid and cr['failure_count']==0 and not cr['missing_evidence']
 assert rp['status']=='PASS_SAMPLE_REPLAY_WITH_LIMITATIONS' and rp['registered_job']==jid and rp['checked_examples']==160 and rp['before_after_records']==320 and rp['mismatch_count']==0
 assert rp['parameters_unchanged'] and rp['inputs_unchanged'] and rp['max_absolute_logit_error']<=1e-4
 assert sp['status']=='PASS_SELECTION_BINDING' and sp['run']==jid and not sp['failures']
 for report in [rp,sp]:
  for rel,digest in report['bound_input_sha256'].items():assert sha(rel)==digest,rel
 m=load(run/'native-exposed-metrics.json');complete=load(run/'training-complete.json');assert complete['global_step']==(30860 if 'fb15k237' in jid else 14900) and complete['current_epoch'] in [19,20]
 assert sha(m['checkpoint'])==m['checkpoint_sha256']==cr['evidence']['training_checkpoint']['checkpoint_sha256']
 assert sha(run/'native-exposed-examples.jsonl')==cr['evidence']['export_sha256']
 for role in ['edit','locality']:close_mapping(m[role],cr['evidence']['recalculated_metrics'][role])
 replay_dir=(R/rec['sample_replay_report']).parent;observations=[json.loads(s) for s in (replay_dir/'replayed-observations.jsonl').read_text().splitlines()]
 assert len(observations)==320 and all(o['rank_exact'] and o['top3_exact'] and o['logit_within_fixed_atol'] for o in observations)
 alias=rp['architecture']['native_alias_restore'];assert alias['strict_root_load'] and alias['cpu_storage_shared'] and alias['child_order']==['ex_model','ex_learner']
 assert alias['effective_base_embedding_sha256']==alias['learner_embedding_checkpoint_sha256']
 limits=['Single seed0; exposed edit_test selects checkpoint; not independent heldout.','Repeated locality probes with workers4, not full unique L-Test or author workers32.','Native CPU alias restoration overwrites base embedding with editor embedding; certify this effective model only.','Fixed160sampled examples, not independent all-score reconstruction.','No complete independent optimizer/validation-forward or independent completed code review claimed.']
 summary=dict(id='KGE-LIT-004-'+job['dataset'],parent='KGE-LIT-004',parent_complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_NATIVE_NUMERICS_WITH_RESTORE_LIMITATIONS',paper_metric_reproduction='DESCRIPTIVE_ONLY_NOT_CERTIFIED',native_metrics=m,article_comparison=cr['evidence']['article_comparison'],training_epochs=20,global_step=complete['global_step'],cpu_checks=cr['checks'],cpu_failures=0,independent_examples=160,before_after_records=320,rank_top3_exact=True,max_target_logit_error=rp['max_absolute_logit_error'],native_restore_semantics=alias,outer_elapsed_s=t['elapsed_s'],limitations=limits)
 prepared.append(dict(jid=jid,run=run,rec=rec,sel=sel,summary=summary,replay_dir=replay_dir))
release=R/'results/KGE-LIT-004';assert release.exists();assert not (release/'summary.json').exists()
# Validate all existing immutable component releases before writing the aggregate.
for component in release.iterdir():
 if component.is_dir() and (component/'manifest-sha256.json').exists():assert_manifest(component)
shutil.copy2(R/'experiments/KGE-LIT-004/experiment.yaml',release/'canonical_before_promotion.yaml')
for item in prepared:
 jid=item['jid'];run=item['run'];dest=release/jid
 if dest.exists():
  assert_manifest(dest)
  prior=load(dest/'summary.json');assert prior['native_metrics']==item['summary']['native_metrics']
  continue
 dest.mkdir()
 paths=[(R/item['rec']['cpu_audit_report'],'independent-cpu-audit-v2.json'),(R/item['rec']['sample_replay_report'],'native-alias-replay-v3.json'),(R/item['sel']['report'],'selection-binding-v3.json'),(run/'terminal.json','terminal.json'),(run/'native-exposed-metrics.json','native-exposed-metrics.json'),(run/'training-complete.json','training-complete.json'),(master,'registered-jobs.json'),(item['replay_dir']/'replayed-observations.jsonl','replayed-observations.jsonl'),(item['replay_dir']/'reconstructed-edit-conditions.jsonl','reconstructed-edit-conditions.jsonl')]
 for src,name in paths:shutil.copy2(src,dest/name)
 (dest/'summary.json').write_text(json.dumps(item['summary'],indent=2)+'\n')
 (dest/'manifest-sha256.json').write_text(json.dumps({str(p.relative_to(dest)):sha(p) for p in sorted(dest.iterdir()) if p.is_file()},indent=2)+'\n')
aggregate=dict(id='KGE-LIT-004',status='PASS_NATIVE_NUMERICS_WITH_RESTORE_LIMITATIONS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=[i['summary'] for i in prepared],cpu_checks=sum(i['summary']['cpu_checks'] for i in prepared),cpu_failures=0,independent_examples=320,before_after_records=640,retained_failures=['Original WN CPU auditor wrong epoch convention FAIL','WN separated-prefix replay v2 FAIL'],limits=prepared[0]['summary']['limitations'])
(release/'summary.json').write_text(json.dumps(aggregate,indent=2)+'\n');shutil.copy2(release/'summary.json',R/'tables/KGE-LIT-004-completed-qualified.json')
for rec in native['jobs']:rec.update(status='completed_audited_qualified',audit_status=aggregate['status'])
native.update(status='completed_audited_qualified',audit=dict(status=aggregate['status'],checks=aggregate['cpu_checks'],failures=0,independent_examples=320,before_after_records=640,evidence_directory='results/KGE-LIT-004',summary_sha256=sha(release/'summary.json'),limitations=aggregate['limits'],retained_failures=aggregate['retained_failures']))
cp=R/'experiments/KGE-LIT-004/experiment.yaml';tmp=cp.with_suffix('.yaml.tmp');tmp.write_text(yaml.safe_dump(native,sort_keys=False));tmp.replace(cp);shutil.copy2(cp,release/'canonical_after_promotion.yaml')
(release/'manifest-sha256.json').write_text(json.dumps({str(p.relative_to(release)):sha(p) for p in sorted(release.rglob('*')) if p.is_file() and p!=release/'manifest-sha256.json'},indent=2)+'\n')
print(json.dumps(dict(status=native['status'],checks=aggregate['cpu_checks'],independent_examples=320,release=str(release))))
