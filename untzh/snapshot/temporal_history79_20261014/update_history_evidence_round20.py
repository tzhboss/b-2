"""Persist source exclusions; preserve fixed40 and all existing truth labels."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/name).read_text(encoding='utf8'))
e=read('representative-history-evidence.json'); sample=read('representative-history-sample.json')
paired=read('paired-truth-drift-bounds-round19.json')
assert sample['sample_n']==len(sample['records'])==40 and sample['population_n']==42851
keys=('gold_truth','original_top1_truth','final_event_truth','reasonable_rank_adjustment','stage')
before={k:tuple(j.get(x) for x in keys) for k,j in e['judgments'].items()}
sources={
 'R20_KIBAKI_BLOG_MAY24_NONMATCH':{
  'url':'https://stopwarsaveearth.blogspot.com/2005_05_24_archive.html',
  'read_status':'actual_relevant_body_read_lines177_182',
  'blog_date':'2005-05-24','copied_article_date':'2005-05-18','caption_event_date':'2005-05-16',
  'linked_article_url':'http://www.metimes.com/articles/normal.php?StoryID=20050518-082750-5456r',
  'evidence_summary':'Copied item reports Wednesday May18 magistrate decision; May16 is court photograph/private suit caption. REUTERS label belongs to photograph, not proven wire authorship. AFP May16 credit elsewhere belongs to separate Lahore women-rights item. Does not recover sampled AFP Story5273848 or support May16 riot.'},
 'R20_KIBAKI_BLOG_MAY16_NONMATCH':{
  'url':'https://stopwarsaveearth.blogspot.com/2005_05_16_archive.html',
  'read_status':'page_opened_and_Kibaki_find_returned_no_matching_text',
  'evidence_summary':'Archive date alone does not identify original article. No matching Kibaki text in returned page; source-route exclusion, not evidence of absence of a May16 event.'},
 'R20_SI_OFFICIAL_PAKISTAN_REPORT':{
  'url':'https://www.internationalesocialiste.org/fileadmin/uploads/si/Documents/Congresses/Athens/SG_Report_XXIII_SI_Congress_Athens_2008aa.pdf',
  'read_status':'actual_official_report_body_read_P30_31_lines953_1014',
  'evidence_summary':'Report gives Bhutto June2007 Geneva speech, SI condemnation of October bombing, December27 assassination and later activities. Does not establish sampled October21 Bhutto/Papandreou phone consultation. No label from silence or organization-level solidarity.'},
 'R20_HRI_ROUTE_ATTEMPTS':{
  'url':'https://www.hri.org/news/greek/ana/2007/07-10-22.ana.html',
  'route_identity':'speculative_archive_path_not_verified_article_identity',
  'read_status':'web_inaccessible_and_both_curl_routes_failed_to_connect',
  'local_https_session_id':32662,'local_https_exit_code':1,'local_https_reported_connect_milliseconds':22562,
  'remote_http_session_id':53569,'remote_http_exit_code':1,'remote_http_reported_connect_milliseconds':130010,
  'evidence_summary':'Original handles polled to natural terminal errors; remote curl absent in ps. No article body or headers received. No execution hard timeout, process restart or truth inference.'}
}
e['sources'].update(sources)
for sid, ids in {'S04':['R20_KIBAKI_BLOG_MAY24_NONMATCH','R20_KIBAKI_BLOG_MAY16_NONMATCH'],'S05':['R20_SI_OFFICIAL_PAKISTAN_REPORT','R20_HRI_ROUTE_ATTEMPTS']}.items():
 j=e['judgments'][sid]
 j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+ids))
 j['round20_source_findings']=[sources[k]['evidence_summary'] for k in ids]
j=e['judgments']['S05']
j['round18_lifetime_candidate_note']='JohnPaulII newly exceeds gold. Conditional on new personal action at query date, paired round19 proves false-rank drift lower>=1 because no old outranker exits. Original gold date independently unvalidated; no certified harmful drift against valid history. Retrospective/reporting semantics unresolved.'
j['round20_inference_correction']='Earlier one-negative-does-not-establish-net-drift wording ignored paired cancellation. Personal-action mechanism lower1 is established; remaining S05 barrier is independent gold support and coder-date semantics.'
e['round20_source_exclusion_progress']={
 'receipt_file':'round20-web-receipts.json','notes_file':'round20-source-notes.md',
 'receipt_n':len(read('round20-web-receipts.json')['results']),'new_source_truth_labels_n':0,
 'sample_ids_targeted':['S04','S05'],'world_reasonable_fraction':None,'world_error_fraction':None,
 'sample_replaced_or_expanded':False,'training':False,'neural_rescoring':False,
 'terminal_fetch_sessions':[32662,53569],'pending_round20_processes':[]}
assert before=={k:tuple(j.get(x) for x in keys) for k,j in e['judgments'].items()}
assert paired['summary']['organization_occurrence_external_positives_only']['certified_positive_ids']==['S38']
assert paired['summary']['personal_action_occurrence_external_positives_only']['certified_positive_ids']==['S33','S38']
(ROOT/'representative-history-evidence.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'source_truth_labels_changed':0,'receipt_n':e['round20_source_exclusion_progress']['receipt_n'],'sample_n':40,'world_error_fraction':None,'pending_round20_processes':[]},ensure_ascii=False))
