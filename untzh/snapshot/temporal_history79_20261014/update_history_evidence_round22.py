"""Persist new source-body leads without silently relaxing exact-event-date semantics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/name).read_text(encoding='utf8'))
e=read('representative-history-evidence.json')
s=read('representative-history-sample.json')
assert s['sample_n']==len(s['records'])==40
keys=('gold_truth','original_top1_truth','final_event_truth','reasonable_rank_adjustment','stage')
old={sid:tuple(j.get(k) for k in keys) for sid,j in e['judgments'].items()}
new={
 'R22_LABOUR_AGREEMENT_BODY':{
  'url':'https://www.vanguardngr.com/2011/07/fg-labour-agreement-that-aborted-the-strike/',
  'read_status':'actual_original_article_body_read_lines64_71',
  'publication_date':'2011-07-23',
  'evidence_summary':'Previously navigation-only route now yields actual body. Recalls prior agreement; July31 negotiations deadline and August1 wage implementation are future conditions, not July26 cooperation. No exact AllAfrica Story15699991 linkage or Obama truth.'},
 'R22_LABOUR_STRIKE_CALLED_OFF_BODY':{
  'url':'https://www.vanguardngr.com/2011/07/labour-calls-off-strike/',
  'read_status':'actual_original_article_body_read_lines64_108',
  'publication_date':'2011-07-20',
  'evidence_summary':'Original body recounts midnight strike cancellation after Abuja negotiations. Ibadan section describes previous-day strike preparation. Does not establish sampled July26 diplomatic cooperation and is not an absence proof.'},
 'R22_GULWE_ALLAFRICA_REPRINT_BODY':{
  'url':'https://buildmartafrica.com/detail-news.php?NEWS_ID=330',
  'read_status':'actual_reprint_body_read_lines27_43',
  'attributed_source':'allafrica.com',
  'displayed_publication_date':'30 Nov,-0001',
  'publication_date_valid':False,
  'same_internal_story_proven':False,
  'candidate_sentence_alignment':{'Kikwete_statement':3,'Magufuli_statement':6},
  'evidence_summary':'Mpwapwa/Gulwe bridge body names both actors speaking; counting consecutive prose sentences gives Kikwete said at3 and Magufuli said at6, matching rawStory35330934 coding positions. Strong candidate original story lead, not internalStoryID proof. Invalid display date and no occurrence day prevent Sep1 truth labels.'},
 'R22_GULWE_MWANANCHI_FORUM_REPRINT':{
  'url':'https://www.jamiiforums.com/threads/rais-kikwete-nimetimiza-ahadi-kwa-asilimia-80.716796/',
  'read_status':'actual_forum_reprint_body_read_lines83_136',
  'forum_post_date':'2014-09-01',
  'claimed_original_source':'Mwananchi',
  'relative_occurrence_word':'juzi',
  'relative_date_interpretation':'day_before_yesterday; Aug30 only if original copy date matches forum date',
  'evidence_summary':'Reprint says Kikwete statement atGulwe happened juzi. Forum timestamp is not necessarily original publication date; originalMwananchi date absent. Earlier-date lead only, no Sept1 gold disproof or support.'},
 'R22_GULWE_PHOTO_INDEX_LEAD':{
  'url':'https://michuzi-matukio.blogspot.com/2014/08/rais-kikwete-azindua-ujenzi-wa-daraja.html',
  'read_status':'search_index_text_available; first_open_header_only_then_reopen_Google_error; not_actual_line_numbered_body',
  'index_publication_date':'2014-08-30',
  'evidence_summary':'Index shows bridge visit with both actors and Aug30post. Kept as lead, not acquired photo-caption evidence or precise occurrence proof.'},
 'R22_IKULU_SPEECH_FETCH':{
  'url':'https://www.ikulu.go.tz/speeches?page=12',
  'read_status':'web_timeout; remoteurllib_SSL_EOF_no_body',
  'execution':read('ikulu-speeches-fetch-round22.json'),
  'original_session_id':11622,'process_exit_code':0,
  'evidence_summary':'Search indexed Dodoma speech text but no page body retrieved. Natural exit0 only confirms captured error record. No truth from search preview or retrieval failure.'},
}
e['sources'].update(new)
for sid,ids in {
 'S40':['R22_LABOUR_AGREEMENT_BODY','R22_LABOUR_STRIKE_CALLED_OFF_BODY'],
 'S27':['R22_GULWE_ALLAFRICA_REPRINT_BODY','R22_GULWE_MWANANCHI_FORUM_REPRINT','R22_GULWE_PHOTO_INDEX_LEAD','R22_IKULU_SPEECH_FETCH'],
}.items():
 j=e['judgments'][sid]
 j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+ids))
 j['round22_source_findings']=[new[k]['evidence_summary'] for k in ids]
j=e['judgments']['S27']
j['raw_final_top1_name']='John Pombe Magufuli'
j['raw_final_top1_is_benchmark_filtered']=True
j['filtered_new_outranker_names']=['Domestic Affairs (Tanzania)','Ministry (Tanzania)']
j['round13_source_note']=j['round13_source_note'].replace('soleafterTop1DomesticAffairs','filteredcompetitorDomesticAffairs (rawfinalTop1isMagufuli)')
j['round22_filter_note']='Raw finalMagufuli already knownandfiltered; new reprint support would not explain filteredgold1to3. The two actualcrossing competitors remainunknown.'
j['rationale']+=' Round22: ActualAllAfrica-attributed Gulwe reprint has matching actor/sentence-order3and6 and place, but invaliddisplaydate/noStoryIDlink. ForumMwananchi copy usesjuzi; keep exactSep1truthunknown and filter distinctions explicit.'
e['judgments']['S40']['rationale']+=' Round22: Actualpreviouslynavigation-only agreement body nowread, datedJuly23; anotheroriginalbodydatedJuly20hasAbujanegotiations/Ibadanstrikepreparation. Theseareearlierreports, not sampledJuly26cooperation.'
e['round22_source_provenance_progress']={
 'receipt_file':'round22-web-receipts.json','notes_file':'round22-source-notes.md',
 'receipt_n':len(read('round22-web-receipts.json')['results']),
 'new_actual_body_source_ids':['R22_LABOUR_AGREEMENT_BODY','R22_LABOUR_STRIKE_CALLED_OFF_BODY','R22_GULWE_ALLAFRICA_REPRINT_BODY','R22_GULWE_MWANANCHI_FORUM_REPRINT'],
 'new_truth_labels_n':0,'sample_replaced_or_expanded':False,
 'world_reasonable_fraction':None,'world_error_fraction':None,
 'training':False,'neural_rescoring':False,'pending_round22_processes':[]}
assert old=={sid:tuple(j.get(k) for k in keys) for sid,j in e['judgments'].items()}
(ROOT/'representative-history-evidence.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(e['round22_source_provenance_progress'],ensure_ascii=False))
