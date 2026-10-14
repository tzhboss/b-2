"""Record actually read external sources and preserve unresolved outcomes."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
p = ROOT/'representative-history-evidence.json'
e = json.loads(p.read_text(encoding='utf-8'))
new_sources = {
 'ABBAS_INDYK_2013': {
  'url':'https://www.business-standard.com/amp/article/news-ians/palestine-seeks-more-us-role-in-peace-talks-113091700936_1.html',
  'title':'Palestine seeks more US role in peace talks', 'publisher':'IANS relaying Xinhua via Business Standard',
  'read_status':'article_body_read', 'publication_date':'2013-09-17','event_date':'2013-09-17',
  'evidence_summary':'Tuesday Abbas met US envoy Martin Indyk; they discussed arrangements for an Obama meeting the following week. Supports gold Consult US Envoy. Does not exclude a separate Netanyahu consultation.'},
 'PAPANDREOU_NONMATCH_2007': {
  'url':'https://archive.patris.gr/articles/119945/64024','title':'Papandreou party-campaign report (nonmatching event)',
  'read_status':'article_body_read_in_previous_goal_turn','publication_date':'2007-10-22',
  'evidence_summary':'Read article concerned the Larissa speech and party campaign, not the sampled Bhutto call or a Tsipras consultation. No truth label from absence.',
  'alternate_source_attempt':'https://www.hri.org/news/greek/ana/2007/07-10-22.ana.html','alternate_source_result':'not accessible via browser tool; speculative archive path, not verified article identity'},
 'EAC_NONMATCH_2014': {
  'url':'https://www.newtimes.co.rw/article/107451/News/eac-states-unveil-new-initiatives-aimed-at-attracting-more-tourists/amp',
  'title':'EAC states unveil new initiatives aimed at attracting more tourists','read_status':'article_body_read','publication_date':'2014-06-02',
  'evidence_summary':'Lenku comments on tourism integration and the article mentions Burundi; the report describes the preceding week, not March31. Cannot support exact sampled date.'},
 'WEN_ANGOLA_2006': {
  'url':'https://www.mfa.gov.cn/gjhdq_676201/gj_676203/fz_677316/1206_677390/xgxw_677396/200606/t20060621_8015169.shtml',
  'title':'温家宝会见安哥拉总理费尔南多','read_status':'official_article_body_read','publication_date':'2006-06-21','event_date':'2006-06-20',
  'evidence_summary':'Explicitly dates the prime-minister meeting to June20 afternoon and reports cooperation agreements already signed that day. Supports Angola-China diplomatic cooperation at country granularity.'},
 'ANGOLA_EXTRADITION_2006': {
  'url':'https://www.mfa.gov.cn/web/wjb_673085/zzjg_673183/tyfls_674667/xwlb_674669/200606/t20060622_7669516.shtml',
  'title':'《中安引渡条约》签署','read_status':'official_article_body_passage_read','publication_date':'2006-06-22','event_date':'2006-06-20',
  'evidence_summary':'Explicitly records June20 signing in Luanda by Chinese foreign minister Li Zhaoxing and Angolan acting foreign minister Georges Chikoti. Independently supports a completed diplomatic cooperation event, not merely an intended agreement.'},
 'G5_SAHEL_APPEAL_2014': {
  'url':'https://www.lorientlejour.com/article/901801/cinq-pays-du-sahel-appellent-lonu-a-intervenir-en-libye-en-accord-avec-lua-sommet-a-nouakchott.html',
  'title':"Cinq pays du Sahel appellent l'ONU à intervenir en Libye en accord avec l'UA (sommet à Nouakchott)",
  'publisher':'AFP via L Orient Le Jour','read_status':'relevant_two_body_paragraphs_read_site_also_displays_subscription_UI',
  'publication_date':'2014-12-19','event_date':'2014-12-19',
  'evidence_summary':'Friday Nouakchott summit final communique of Chad, Mali, Niger, Mauritania and Burkina Faso appealed to the UN Security Council for an international force. Supports sample gold Mauritania at country-representative granularity; no proof concerning Mogherini.'},
 'SOLANA_JAN2007_FETCH': {
  'url':'https://www.un.org/unispal/document/auto-insert-197355/','title':'Chronological Review of Events/January 2007',
  'read_status':'search_result_only_open_HTTP403','evidence_summary':'No full chronology read; search result not sufficient to label Solana-Livni or Solana-Mogherini consultation on Jan16.'},
 'KARIMOV_VISITS_FETCH': {
  'url':'https://www.koreapost.com/news/articleView.html?idxno=32996','title':'Ambassador Fen of Uzbekistan helps Korea, Uzbekistan brother countries',
  'read_status':'search_result_only_open_HTTP403','evidence_summary':'Snippet lists official visit years excluding2009, but no body read. Even an official-visit list would not exclude every informal visit. No negative label.'},
 'ANANDA_RAW_CONTEXT': {
  'url':None,'local_file':'all-sample-coded-context.json','read_status':'original_coded_metadata_read_not_news_body',
  'evidence_summary':'S03 raw Ananda source is encoded Myanmar and International Religious/Buddhist. Indonesian racing-driver search results must not be assumed to identify this raw entity. Raw classification itself may be erroneous; exact identity unresolved.'}
}
e['sources'].update(new_sources)
def judgment(stage, ids, rationale, gold='unknown', before='unknown'):
 return {'stage':stage,'final_event_truth':'unknown','original_top1_truth':before,'gold_truth':gold,'reasonable_rank_adjustment':'unknown','evidence_ids':ids,'rationale':rationale}
e['judgments'].update({
 'S02':judgment('detailed_source_review',['ABBAS_INDYK_2013'],'Gold US envoy consultation supported on exact date. Future Obama meeting is distinct. Neither Obama nor Netanyahu same-day consultation is exhaustively excluded.','supported_at_query_semantic_granularity'),
 'S03':judgment('coded_context_identity_review',['ANANDA_RAW_CONTEXT'],'Do not substitute the Indonesian racing driver for the Myanmar Buddhist actor code; final lawyer top1 is unchanged and exact article/identity remain unknown.'),
 'S05':judgment('detailed_source_review',['PAPANDREOU_NONMATCH_2007'],'Opened campaign article is not the sampled Bhutto call. HRI guessed archive path inaccessible; neither supports nor contradicts final Tsipras.'),
 'S09':judgment('detailed_source_review',['EAC_NONMATCH_2014'],'Original same Story33087140 sentence5 encodes five countries. June2 article is a different date. Multiple recorded answers do not prove which real-world candidates should outrank Tanzania on March31.'),
 'S11':judgment('detailed_source_review',['WEN_ANGOLA_2006','ANGOLA_EXTRADITION_2006'],'Completed China-Angola cooperation on June20 independently supported. Final Portugal cooperation not established or excluded; gold truth alone is insufficient to establish whole-rank error.','supported_at_query_semantic_granularity', 'supported_at_query_semantic_granularity'),
 'S30':judgment('detailed_source_attempt_blocked',['SOLANA_JAN2007_FETCH'],'Official chronology403. No truth label from snippets or assumption that a future officeholder could not hold any earlier consultation.'),
 'S32':judgment('detailed_source_attempt_blocked',['KARIMOV_VISITS_FETCH'],'Visit-year search result is not full article and not exhaustive informal travel. Exact Oct13 host visit unknown.'),
 'S37':judgment('detailed_source_review',['G5_SAHEL_APPEAL_2014'],'AFP same-day communique supports Mauritania UNSC appeal. No evidence to exclude a separate Mogherini appeal; final truth unknown.','supported_at_query_semantic_granularity')
})
e['source_context_scan_cost_note']={'continuation_2014_seconds':2.816116,'initial_2005_2013_total_seconds':None,'scope':'zip-only initial scanner failed at uncompressed2014.tab after saving2005-2013; resumed2014 only. Exact initial duration unavailable; do not report continuation as full scan cost.'}
e['judgments']['S33']['cached_filter_note'] = 'Xi is a recorded positive already filtered from Qin rank; his supported raw Top1 cannot cause or justify Qin filtered rank3->62.'
e['judgments']['S39']['conditional_missing_alternative_filter_explanation'] = json.loads((ROOT/'sample-filter-interpretation.json').read_text(encoding='utf-8'))['records'][38]['semantic_alternative_filter_diagnostic']
e['judgments']['S39']['cached_filter_note'] = 'Cached average-rank evaluator and scores show Iraq strictly above China, unique raw Top1, final filtered gold rank2. If Iraq is treated as an additional semantic positive, filtering it yields rank1 and zero gold drift; original China event validity remains unknown.'
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('sources',len(e['sources']),'explicit judgments',len(e['judgments']))
