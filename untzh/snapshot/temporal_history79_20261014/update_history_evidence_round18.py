"""Persist Pope lifespan context and clarify existing uncertainty; no new world labels."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
e=read('representative-history-evidence.json')
labelkeys=('gold_truth','final_event_truth','reasonable_rank_adjustment')
before={k:tuple(j[x] for x in labelkeys) for k,j in e['judgments'].items()}
e['sources'].update({
 'R18_POPE_OFFICIAL_DEATH_AND_POSTHUMOUS_TEXT':{
  'url':'https://www.vatican.va/content/john-paul-ii/en/angelus/2005/documents/hf_jp-ii_reg_20050403_divina-misericordia.html',
  'read_status':'actual_official_body_read_lines31_38',
  'death_date':'2005-04-02','text_read_date':'2005-04-03',
  'evidence_summary':'Official dated page confirms JohnPaulII death on April2 and that ArchbishopSandri read previously prepared papal text April3. Establishes both lifespan and an actual posthumous-publication example. A dated report of words cannot be equated with new personal utterance. Does not support the 2007consult or 2008expel sample candidates.'},
 'R18_POPE_RAW_POSTHUMOUS_IDENTITY':{
  'artifact':'pope-identity-context-round18.json','read_status':'five_existing_raw2005_postdeath_rows_read',
  'evidence_summary':'Catholic international-religious actor identity. April3 ReutersStory6550586sentence4 codes JohnPaulII MakeVisitMexico and MexicoHostVisitJohnPaulII after death. Metadata proves such coding exists, but lacks original sentence and exact source URL. Other rows are actions toward deceased Pope; those are not automatically invalid.'},
 'R18_POPE_MEXICO_REUTERS_RETROSPECTIVE':{
  'url':'https://www.lanueva.com/nota/2005-4-4-9-0-0-el-masivo-adios-en-latinoamerica',
  'read_status':'actual_Reuters_attributed_body_via_LaNueva_read_lines36_56',
  'url_date':'2005-04-04',
  'evidence_summary':'Contemporary obituary/mourning article recalls five earlier Mexico visits. Related Reuters context, not a proven identity match with internalStory6550586 or source-language translation. No exact raw-record truth label.'},
 'R18_POPE_MEXICO_JORNADA_RETROSPECTIVE':{
  'url':'https://www.jornada.com.mx/2005/04/03/index.php?article=018n1pol&section=politica',
  'read_status':'actual_original_newspaper_body_read_lines17_22',
  'publication_date':'2005-04-03',
  'evidence_summary':'Reports Fox mourning the Pope and explicitly recalls last Mexico visit in2002. Confirms retrospective visit reporting on April3, not internalReutersStory identity.'},
 'R18_BHUTTO_PAPANDREOU_SOURCE_SEARCH':{
  'receipt_file':'round18-web-receipts.json','read_status':'searches_no_exact_body; BirkbeckPDF_open_failed',
  'evidence_summary':'English/Greek searches yielded campaigning/other-date and secondary snippets. No independent exactOctober21consultation body recovered. S05gold remains unknown. AnandaApril18identity search likewise no dated article identity.'}
})
crossings=read('fixed40-new-outrankers.json')
bounds=read('false-rank-drift-bounds-round10.json')
records=[]
for sampleid in ['S05','S08']:
 r=next(r for r in crossings['records'] if r['sample_id']==sampleid)
 c=next(c for c in r['new_outrankers'] if c['fact'][0]=='John Paul II')
 b=next(r for r in bounds['records'] if r['sample_id']==sampleid)
 records.append({**c,'sample_id':sampleid,'gold_fact':r['gold_fact'],
  'conditional_personal_occurrence_truth':'contradicted_after_death',
  'reported_historical_content_truth':'unknown','unconditional_temporal_negative':False,
  'gold_supported':False,'before_raw_outrankers_n':b['raw_before_above_gold_n'],
  'whole_error_rank_drift_newly_proven':False,
  'source_ids':['R18_POPE_OFFICIAL_DEATH_AND_POSTHUMOUS_TEXT','R18_POPE_RAW_POSTHUMOUS_IDENTITY']})
 j=e['judgments'][sampleid]
 j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+records[-1]['source_ids']))
 j['round18_lifetime_candidate_note']='Existing JohnPaulII entity newly exceeds gold. New personal2007/2008action is incompatible with2005death; dated retrospective report remains unknown. Gold unverified and one conditional negative cannot establish positive whole false-rank drift.'
assert before=={k:tuple(j[x] for x in labelkeys) for k,j in e['judgments'].items()}
write('pope-crossing-interpretation-round18.json',{'sample_n':40,'records':records,
 'new_conditional_pairwise_components_n':2,'new_unconditional_negative_n':0,'new_whole_error_rank_drift_n':0,
 'raw_metadata_scan_seconds':read('pope-identity-context-round18.json')['seconds'],
 'training':False,'neural_rescoring':False})
# Correct the old prose's raw/filtered count without changing the original score data.
old=read('lifetime-crossing-interpretation-round12.json')
s33=next(r for r in bounds['records'] if r['sample_id']=='S33')
old['note']='S33 has3 raw before outrankers, including independently supported Xi already benchmark-filtered; filtered before has2. One conditional Chiang negative cannot force positive total false-rank drift. S22/S29 golds remain unknown.'
old['round18_count_clarification']={'raw_before':s33['raw_before_above_gold_n'],'filtered_before':s33['filtered_before_above_gold_n'],
 'reason':'Prior prose mislabeled2 as raw. Cached counts are unchanged; counting Xi as true can reduce possible-false upper but requires that positive interpretation.'}
write('lifetime-crossing-interpretation-round12.json',old)
e['round18_lifetime_artifact']='pope-crossing-interpretation-round18.json'
write('representative-history-evidence.json',e)
print(json.dumps({'new_conditional_components':2,'new_whole_errors':0,'truth_labels_changed':0,'sample_n':40}))
