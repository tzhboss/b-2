"""Build distinct real-result tables from saved audited evidence; unavailable measurements remain null."""
from pathlib import Path
import json,csv,datetime,yaml,hashlib
R=Path(__file__).resolve().parents[1]
def load(name):return json.loads((R/name).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):
 p=R/'tables'/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def write_csv(name,rows):
 with (R/'tables'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[]
inc=load('tables/KGE-LIT-001-completed.json');times=load('tables/incde-native-training-times.json')
# The time receipt schema is read explicitly; metrics come from audited summary.
if isinstance(times,dict):
 times=times.get('runs',times.get('records',times))
time_index={x['id']:x for x in times} if isinstance(times,list) else times
for x in inc['runs']:
 m=x['native_final'];t=time_index[x['id']]
 seconds=t['native_train_loop_s']
 assert seconds is not None,'native IncDE time field requires explicit source schema check'
 rows.append(dict(method='IncDE',dataset=x['dataset'],seed=3407,protocol_group='native_IncDE_'+x['dataset'],scope='complete5snapshots_native',MRR=m['mrr'],Hits1=m['hits1'],Hits3=m['hits3'],Hits10=m['hits10'],native_training_seconds=seconds,full_process_seconds=x['outer_elapsed_s'],max_gpu_memory_MB=None,audit_status='PASS_WITH_LIMITATIONS',independent_rank_replays=x['independent_rank_replays'],source_artifact='results/KGE-LIT-001/summary.json',primary_source='https://doi.org/10.1609/aaai.v38i8.28722',limitations='single seed; historical candidate/filter; sampled checkpoint inference; memory peak not measured'))
for x in load('tables/KGE-LIT-002-completed.json'):
 m=x['native_final_metrics'];rows.append(dict(method=x['method'],dataset=x['dataset'],seed=x['seed'],protocol_group='native_FB_CKGE_same_raw_bytes',scope='complete5snapshots_native',MRR=m['mrr'],Hits1=m['hits1'],Hits3=m['hits3'],Hits10=m['hits10'],native_training_seconds=x['native_training_seconds'],full_process_seconds=x['full_process_seconds'],max_gpu_memory_MB=None,audit_status=x['status'],independent_rank_replays=x['posttrain_sample_replay_exact'],source_artifact='results/KGE-LIT-002/'+x['id']+'/summary.json',primary_source='https://doi.org/10.24963/ijcai.2024/243' if x['method']=='FastKGE' else 'https://github.com/bigdata-inha/STARK',limitations='; '.join(x['limitations'])+'; memory peak not measured'))
x=load('tables/KGE-LIT-003-qualified.json');m=x['native_final']['native_round3'];rows.append(dict(method='SAGE',dataset='FACT',seed=55,protocol_group='released_SAGE_FACT_multiset_only',scope='complete5snapshots_released_fixed_dimension_schedule',MRR=m['whole_mrr'],Hits1=m['whole_hits1'],Hits3=m['whole_hits3'],Hits10=m['whole_hits10'],native_training_seconds=x['native_train_loop_seconds'],full_process_seconds=x['outer_elapsed_s'],max_gpu_memory_MB=None,audit_status='FAIL_RETAINED_NUMERICS_QUALIFIED',independent_rank_replays=x['sampled_rank_replays'],source_artifact='results/KGE-LIT-003/summary.json',primary_source='https://arxiv.org/abs/2508.11347',limitations='original producer metadata integrity FAIL; fixed dimensions180/210/250/270/290; single seed; not clean reproduction; memory peak not measured'))
x=load('tables/KGE-LIT-006-completed.json');m=x['native_final'];rows.append(dict(method='LKGE_clean_validation',dataset='FACT',seed=55,protocol_group='clean_valid_adaptation_LKGE_FACT',scope='complete5snapshots_declared_adaptation',MRR=m['whole_mrr'],Hits1=m['whole_hits1'],Hits3=m['whole_hits3'],Hits10=m['whole_hits10'],native_training_seconds=x['native_training_seconds'],full_process_seconds=x['outer_elapsed_s'],max_gpu_memory_MB=None,audit_status=x['status'],independent_rank_replays=x['independent_rank_replays'],source_artifact='results/KGE-LIT-006/summary.json',primary_source='https://doi.org/10.1609/aaai.v37i4.25539',limitations='valid.txt replaces upstream test.txt selection; IDs/filter change; single seed; no clean method advantage claim; memory peak not measured'))
jobs_by_method={}
for eid in ['001','002','003','006']:
 for job in load('configs/experiments/KGE-LIT-'+eid+'-jobs.json'):
  jobs_by_method[job['method']]=job
for row in rows:
 method='LKGE_clean_split' if row['method']=='LKGE_clean_validation' else row['method']
 job=jobs_by_method[method]
 row.update(source_commit=job.get('source_commit'),training_complete=True,split='validation selects checkpoint; official test metrics; actual source split lineage in independent audit',filtering='native historical query-snapshot candidates and truth filters; double argsort',initialization='registered author default; seed'+str(row['seed'])+'; checkpoint/source byte lineage in audit',locality_metric=None)
write('continual_kge_real_results.json',dict(utc=now,rows=rows,not_a_mixed_SOTA_ranking=True,missing_measurement_policy='null, not0; no per-process GPU memory peak was captured',comparison_limit='Only FastKGE/STARK have exactly the same raw split bytes here; all tables preserve declared protocols.'))
write_csv('continual_kge_real_results.csv',rows)
# Include only completed/reviewed editor components; native running metrics never promoted here.
edit=[]
for jid in ['kgeditor-e-fb15k237-s0','kgeditor-e-wn18rr-s0']:
 p=R/'results/KGE-LIT-004'/jid/'summary.json'
 if not p.exists():continue
 x=json.loads(p.read_text());m=x['native_metrics'];edit.append(dict(method='KGEditor',dataset=m['config']['data_type'],seed=0,protocol='native_exposed_shared_embedding_restore',training_epochs=20,Succ1=m['edit']['hits1_after'],Succ3=m['edit']['hits3_after'],ER_roc=m['edit']['article_ER_roc'],RK3=m['locality']['article_RK_at3'],RK_roc=m['locality']['article_RK_roc'],edit_count=m['edit']['count'],locality_exposures=m['locality']['count'],locality_unique_rows=m['locality']['unique_source_rows'],sampled_edit_queries=32,sampled_locality_queries=128,max_logit_error=x['max_target_logit_error'],full_process_seconds=x['outer_elapsed_s'],max_gpu_memory_MB=None,audit_status=x['status'],source_artifact=str(p.relative_to(R)),primary_source='https://arxiv.org/abs/2301.10405',limitations='test reused for selection; repeated locality probes; native CPU shared embedding restore changes effective base weights; no clean heldout or mixed architecture superiority claim'))
write('kgeditor_exposed_real_results.json',dict(utc=now,rows=edit,parent_canonical_status=yaml.safe_load((R/'experiments/KGE-LIT-004/experiment.yaml').read_text())['status'],not_heldout=True,not_continual_KGE_metrics=True))
if edit:write_csv('kgeditor_exposed_real_results.csv',edit)
print(json.dumps(dict(continual_rows=len(rows),reviewed_editor_rows=len(edit),utc=now)))
