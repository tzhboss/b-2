"""Record primary Ananda date evidence without identifying the sampled ambiguous actor."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
e=read('representative-history-evidence.json')
sample=read('representative-history-sample.json')
assert sample['sample_n']==len(sample['records'])==40
keys=('gold_truth','original_top1_truth','final_event_truth','reasonable_rank_adjustment','stage')
old={k:tuple(j.get(x) for x in keys) for k,j in e['judgments'].items()}
e['sources']['R21_ANANDA_ANTARA_PRIMARY_CAPTION']={
 'url':'https://www.antarafoto.com/id/view/30863/pelatihan-mengemud',
 'english_url':'https://www.antarafoto.com/view/30863/training',
 'read_status':'actual_primary_Indonesian_caption_read_lines13_16; English_translation_read_then_followed_original_language',
 'event_date':'2008-04-17','caption_timestamp':'2008-04-17 18:26 WIB',
 'actor_names':['Ananda Mikola','Moreno Soeprapto','Tinton Soeprapto'],
 'evidence_summary':'ANTARA/Widodo original caption explicitly names racers at KPK building Jakarta Thursday17April, giving press information about pursuit/surveillance driver training for KPK investigators. English caption auto-translation has errors, so Indonesian original used. Supports this dated event and named racer only, not sampled Apr18 Ananda/PoliceIndonesia or the Myanmar/Buddhist normalization.'}
e['sources']['R21_ANANDA_JAWAPOS_REPRINT_BODY']={
 'url':'https://morenosoeprapto01.wordpress.com/2013/09/20/moreno-soeprapto-ajari-ngebut-penyidik-kpk/',
 'read_status':'actual_reprinted_body_read_lines7_18; not original JakartaPost',
 'reprint_date':'2013-09-20','claimed_original_source':'Jawa Pos, 18 April 2008',
 'identified_event_date':'2008-04-17',
 'evidence_summary':'Body names AnandaMikola, Moreno and Tinton visiting KPK yesterday, and quotes Ananda about driving training. April18 original-source attribution plus yesterday aligns with primary ANTARA April17 caption. Does not identify JakartaPost Story12471808/sentence6, establish statement directed to police, or rule out another Apr18 statement.'}
e['sources']['R21_ANANDA_ICW_FETCH_ATTEMPTS']={
 'urls':['https://antikorupsi.id/id/article/ananda-moreno-ajari-ngebut-penyidik-kpk','https://antikorupsi.org/id/article/ananda-moreno-ajari-ngebut-penyidik-kpk'],
 'read_status':'search_returned_text_but_both_direct_opens_404; not treated_as_read_original_body',
 'evidence_summary':'Search text is a useful discovery pointer; actual underlying ICW page could not be read. Source conclusions rely on actual ANTARA caption and actual WordPress reprint, not preview.'}
e['sources']['R21_NAZOURI_SEP3_ARRIVAL_BODY']={
 'url':'https://www.aljazeera.net/news/2014/9/3/رئيس-الأركان-الليبي-المعين-يزور-مصر',
 'read_status':'actual_Arabic_news_body_read_lines73_80',
 'identified_arrival_date':'2014-09-03',
 'evidence_summary':'Body says Nazouri arrived Cairo WednesdaySeptember3, planned meetings during visit, and recalls priorAugust26meetingwithHegazy. It does not explicitly date Sobhi consultation to September5 or rule out a September5 consultation. AllAfrica Story35375931 remains unrecovered; stayunknown.'}
e['sources']['R21_HRI_ARCHIVE_FETCH']={
 'artifact':'hri-archive-fetch-execution-round21.json','read_status':'archive_discovery_unavailable_no_snapshot_or_body_received',
 'availability_request':read('hri-archive-fetch-execution-round21.json'),
 'availability_session_id':67797,'availability_process_exit_code':0,
 'availability_note':'Python caught and saved network error; exit0 confirms bookkeeping, not successful retrieval.',
 'cdx_attempt_note':'Web open inaccessible. Remote curl command had URL ampersands expanded by remote shell; sshexit0 is not curlexit0. stderr reports connection failure to web.archive.org:443 after728ms. Treat this malformed CDX attempt as failed, not a valid empty archive search. Separate urllib archive.org availability request used full encoded query correctly.',
 'evidence_summary':'No new accessible ANA original article. Original availability handle polled to terminal; remote ps had no Python. Network failure is not an event-negative label.'}
for sid,ids in {
 'S03':['R21_ANANDA_ANTARA_PRIMARY_CAPTION','R21_ANANDA_JAWAPOS_REPRINT_BODY','R21_ANANDA_ICW_FETCH_ATTEMPTS'],
 'S16':['R21_NAZOURI_SEP3_ARRIVAL_BODY'],
 'S05':['R21_HRI_ARCHIVE_FETCH'],
}.items():
 j=e['judgments'][sid]
 j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+ids))
 j['round21_source_findings']=[e['sources'][k]['evidence_summary'] for k in ids]
j=e['judgments']['S03']
j['related_article_body_review_completed']=True
j['sampled_actor_identity_resolved']=False
j['round21_identity_note']='AnandaMikola is now linked to a precise nearby-date driving-training press event by primary ANTARA, not merely a same-name sports result. Still no originalJakartaPost Story12471808 linkage; MyanmarBuddhist and racer identities must not be collapsed. No exactApr18gold or finalLawyerIndonesia label.'
j['rationale']+=' Round21: Read primary ANTARA Indonesian caption dates named AnandaMikola media event at KPK to Apr17; actual JawaPos reprint attributed Apr18 uses yesterday. This provides a concrete alternative actor/event lead but no sampledJakartaPost identity or Apr18occurrence. Gold and final lawyer remainunknown.'
e['round21_source_provenance_progress']={
 'receipt_file':'round21-web-receipts.json','notes_file':'round21-source-notes.md',
 'receipt_n':len(read('round21-web-receipts.json')['results']),
 'new_source_truth_labels_n':0,'sample_replaced_or_expanded':False,
 'related_source_body_review_n':sum(j['stage']=='detailed_source_review' or j.get('related_article_body_review_completed',False) for j in e['judgments'].values()),
 'coded_identity_pending_ids':['S03'],
 'world_reasonable_fraction':None,'world_error_fraction':None,
 'training':False,'neural_rescoring':False,'pending_round21_processes':[]}
assert old=={k:tuple(j.get(x) for x in keys) for k,j in e['judgments'].items()}
assert e['round21_source_provenance_progress']['related_source_body_review_n']==40
(ROOT/'representative-history-evidence.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(e['round21_source_provenance_progress'],ensure_ascii=False))
