"""Record additional exact-date source reviews without inferring negatives from absence."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
path = ROOT / 'representative-history-evidence.json'
e = json.loads(path.read_text(encoding='utf-8'))
e['sources'].update({
 'PANDOR_MALEMA_TUT': {
  'url': 'https://mg.co.za/news/south-africa/2009-02-18-pandor-malema-hold-talks-on-tut-strike/',
  'title': 'Pandor, Malema hold talks on TUT strike', 'publisher': 'SAPA via Mail & Guardian',
  'read_status': 'contemporaneous_wire_article_body_read', 'publication_date': '2009-02-18', 'event_date': '2009-02-17',
  'evidence_summary': 'Wednesday article explicitly says Pandor and ANCYL leader Malema held closed-door talks in Boksburg Tuesday over the university strike and league education programme. It describes Malema talking to reporters afterwards, not an explicit consultation by Pandor with Media (Africa).',
  'mapping': 'Organization represented by its leader maps to ANC Youth League at broad query granularity. Internal SAPA Story ID not linked.'},
 'FRADKOV_STOLTENBERG_TIMELINE': {
  'url': 'https://www.dagbladet.no/nyheter/slik-blir-den-historiske-delelinja/65079443',
  'title': 'Slik blir den historiske delelinja', 'read_status': 'retrospective_article_body_read',
  'publication_date': '2010-04-27', 'historical_event_date': '2007-06-08',
  'evidence_summary': 'NTB-attributed timeline names both prime ministers announcing June 8 2007 agreement on a 70-kilometre boundary in outer Varangerfjord. This supports joint diplomatic activity, but does not itself explicitly date the Consult meeting; keep that exact relation label unresolved.'},
 'FRADKOV_NETTAVISEN_FETCH': {
  'url': 'https://www.nettavisen.no/artikkel/endelig-en-grenseavtale/s/12-95-1098410',
  'title': 'Endelig en grenseavtale', 'read_status': 'restricted_fetch_no_article_body',
  'evidence_summary': 'Search preview describes meeting but actual body fetch failed; no exact consultation truth label from preview alone.'},
 'CHEN_KYRGYZ_SPEAKER_JAN10': {
  'url': 'https://rw.china-embassy.gov.cn/gnzyxw/201201/t20120111_5995940.htm',
  'title': '胡锦涛主席特使陈至立会见吉尔吉斯斯坦议长叶延别科夫并出席中吉建交20周年招待会',
  'read_status': 'official_article_body_read', 'publication_date': '2012-01-11', 'event_date': '2012-01-10',
  'evidence_summary': 'Official report dates Chen meeting the Kyrgyz speaker in Bishkek to January 10. It is a different country/person from the sampled Kazakhstan legislature query and does not establish that query or rule out every other meeting that day.'},
 'CHEN_KAZAKH_MAMI_PREVIEW': {
  'url': 'https://news.cntv.cn/20120111/125747.shtml', 'title': '陈至立会见哈萨克斯坦议会上院议长马米',
  'read_status': 'search_body_preview_only_open_page_without_relevant_article_text',
  'evidence_summary': 'Search preview dates the Mami meeting to January 9. Opened page lacked the relevant article text; not upgraded to verified original-article identity or strict gold contradiction.'},
 'SOLANA_MIDEAST_SCHEDULE_JAN16': {
  'url': 'https://agenceurope.eu/fr/bulletin/article/9345/6/javier-solana-part-au-proche-orient-ce-mercredi',
  'title': 'Javier Solana part au Proche-Orient ce mercredi', 'publisher': 'Agence Europe',
  'read_status': 'contemporaneous_wire_article_body_read', 'publication_date': '2007-01-17', 'dateline': '2007-01-16',
  'evidence_summary': 'January 16 Brussels dateline describes departure January 17 and planned Jerusalem meetings with Olmert and Livni on Sunday January 21. Scheduled future visit is distinct from an actual January 16 consultation. Does not exclude separate phone contact or prove linkage to AP Story ID.'},
 'KIKWETE_MBEYA_FETCH': {
  'url': 'https://www.thehabari.com/ikulu-yamtumia-rambirambi-rc-mbeya-vifo-vya-watu-10/',
  'title': 'Ikulu Yamtumia Rambirambi RC Mbeya Vifo vya Watu 10', 'read_status': 'inaccessible_no_body_read',
  'evidence_summary': 'September 1 search preview concerns condolences after August 29 crash. Body inaccessible; report date alone not an exact event date.'},
 'KIKWETE_MPWPWA_FETCH': {
  'url': 'https://mpwapwahabari.blogspot.com/2014/09/', 'title': 'Mpwapwa news September 2014 archive',
  'read_status': 'redirected_to_captcha_no_body_read', 'evidence_summary': 'Potential tour report could not be read. No gold or Magufuli statement truth label from the failed fetch.'},
 'HUNGARIAN_YEARBOOK2009_FETCH': {
  'url': 'https://2010-2014.kormany.hu/download/e/64/00000/magyar_kulpolitikai_evkonyv_2009.pdf',
  'title': 'Hungarian foreign policy yearbook 2009', 'read_status': 'HTTP502_no_body_read',
  'evidence_summary': 'Search preview appears to corroborate September 17 ambassador summons. Actual official PDF fetch failed; existing read SITA evidence retained without claiming an official-yearbook body review.'}
})
unknown = {'final_event_truth': 'unknown', 'original_top1_truth': 'unknown',
           'gold_truth': 'unknown', 'reasonable_rank_adjustment': 'unknown'}
e['judgments']['S24'] = {**unknown, 'stage': 'detailed_source_review',
 'gold_truth': 'supported_at_query_semantic_granularity', 'evidence_ids': ['PANDOR_MALEMA_TUT'],
 'rationale': 'Tuesday February 17 closed-door consultation with youth-league leader explicitly confirmed. His later press comments do not confirm the final Pandor/Media(Africa) consultation or exclude one.'}
e['judgments']['S36'] = {**unknown, 'stage': 'detailed_source_review',
 'evidence_ids': ['FRADKOV_STOLTENBERG_TIMELINE', 'FRADKOV_NETTAVISEN_FETCH'],
 'rationale': 'June 8 joint boundary agreement corroborated; exact Consult event requires actual meeting text. Bortnikov consultation with Fradkov not excluded by Stoltenberg activity.'}
e['judgments']['S31'] = {**unknown, 'stage': 'detailed_source_review',
 'evidence_ids': ['CHEN_KYRGYZ_SPEAKER_JAN10', 'CHEN_KAZAKH_MAMI_PREVIEW'],
 'rationale': 'Read official January10 report concerns Kyrgyz speaker. Sample is Kazakhstan legislature in Astana; relevant Mami meeting preview January9 lacks independently read source text and exact story mapping. Avoid declaring sample date wrong solely from another meeting.'}
e['judgments']['S30'] = {**unknown, 'stage': 'detailed_source_review',
 'evidence_ids': list(dict.fromkeys(e['judgments']['S30']['evidence_ids'] + ['SOLANA_MIDEAST_SCHEDULE_JAN16'])),
 'rationale': 'Read contemporary schedule dates identified Livni meeting to January21, not January16 news dateline. AP story context may encode planned meeting as Consult. Exact story identity and separate January16 phone contact unresolved; final Mogherini consultation unknown.'}
e['judgments']['S27'] = {**unknown, 'stage': 'detailed_source_attempt_blocked',
 'evidence_ids': ['KIKWETE_MBEYA_FETCH', 'KIKWETE_MPWPWA_FETCH'],
 'rationale': 'Two potential bodies inaccessible; snippets/September report dates cannot establish exact dated Kikwete or Magufuli statement. No negative label from failed fetching.'}
path.write_text(json.dumps(e, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(ROOT / 'round4-additional-source-notes.md').write_text('''# 补充同日会谈与日期核对

S24：SAPA经Mail & Guardian的2009-02-18正文明确周二（17日）Pandor与青年团领导Malema在Boksburg闭门磋商，支持原gold。报道是Malema向记者谈话，不足以确认最终预测Pandor与Media(Africa)磋商。原gold独立同日语义支持现在6条。

S36：读到NTB回顾时间线，确认2007-06-08两国总理共同宣布边界协议，但当日具体会谈正文未取得，Consult标签未升级。S31：官方1月10日会见是吉尔吉斯斯坦议长；哈萨克斯坦马米会见的1月9日仅有搜索正文预览，打开页面没取得相关文字，不能直接判gold日期错。S30：同月通讯说明1月16日报道的是17日出发、21日拟会见Livni，未来行程不能当当日已经会谈；其他电话仍未排除。

S27的两条当地报道正文获取失败；匈牙利官方年鉴PDF获取502，未宣称读过其正文。失败和未提及均不当负证据。所有40条原样保留，最终Top1支持2条/证伪0条/未定38条；错误成分见证仍为S38一条，未因增加原gold支持自动增加错误计数。
''', encoding='utf-8')
print('ADDITIONAL_SOURCE_REVIEW_SAVED')
