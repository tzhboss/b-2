"""Persist read round5 historical sources; absence never labels a false event."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
p = ROOT / 'representative-history-evidence.json'
e = json.loads(p.read_text(encoding='utf-8'))
e['sources'].update({
 'LANG_BEIJING_AP_EXCERPT': {
  'url': 'https://adamcathcart.com/2009/11/06/frances-nordpolitik/',
  'title': 'France’s Nordpolitik: Jack Lang, French Communists, and French-North Korean Relations',
  'publisher': 'Adam Cathcart historical commentary quoting AP via Taiwan News',
  'publication_date': '2009-11-06', 'event_date': '2009-11-05',
  'read_status': 'dated_commentary_body_with_attributed_AP_excerpt_read_not_original_AP_body',
  'evidence_summary': 'Dated November6 commentary reproduces an AP opening reporting French North Korea envoy Jack Lang arriving in Beijing Thursday, en route to Pyongyang. Thursday of this week is November5. Supports French envoy visit to China at country-level Host semantic granularity. Does not establish every French envoy itinerary or whether another envoy visited The Hague.',
  'mapping': 'Generic Envoy(France) may represent multiple people; no internal Story ID link asserted. AP attribution retained; source is the read excerpt, not independently fetched original AP.'},
 'LANG_TAIWAN_FETCH': {
  'url': 'https://www.etaiwannews.com/etn/news_content.php?cate_img=logo_world&cate_rss=WORLD_eng&id=1100413&lang=eng_news',
  'read_status': 'timeout_no_original_body_read',
  'alternative_url': 'https://www.taiwannews.com.tw/news/1100413',
  'alternative_read_status': 'inaccessible_no_original_body_read'},
 'LEE_MEDVEDEV_PLANNED_NOV10': {
  'url': 'https://www.dailynk.com/english/north-korea-in-g20-spotlight/',
  'title': 'North Korea in G20 Spotlight', 'publisher': 'Daily NK / Chris Green',
  'publication_date': '2010-11-10', 'event_date': '2010-11-10',
  'read_status': 'contemporaneous_article_body_read',
  'evidence_summary': 'Article explicitly says Lee and Medvedev are holding a bilateral meeting this afternoon. This supports a planned same-day meeting for Express intent to meet or negotiate; no substitution of Consult relation. Japan is not independently confirmed by this source.',
  'mapping': 'Same named leaders and date. Asia Pulse original Story ID not linked.'},
 'LEE_JOONGANG_FETCH': {
  'url': 'https://joongangdaily.joins.com/article/view.asp?aid=2928246',
  'title': 'Lee Meets Medvedev in First of Many Bilaterals',
  'read_status': 'timeout_then_inaccessible_no_body_read'},
 'LEE_HERALD_FETCH': {
  'url': 'https://www.koreaherald.com/national/Detail.jsp?newsMLId=20101109001018',
  'title': 'Lee to hold separate talks with G20 leaders',
  'read_status': 'HTTP404_no_body_read',
  'alternative_url': 'https://www.koreaherald.com/view.php?ud=20101109001018',
  'alternative_read_status': 'inaccessible_no_body_read'},
 'NIGERIA_RIOT_AFP_JUNE18': {
  'url': 'https://www.ndtv.com/world-news/36-dead-in-nigeria-church-attacks-rioting-488632',
  'title': '36 dead in Nigeria church attacks, rioting', 'publisher': 'AFP via NDTV',
  'publication_date': '2012-06-18', 'identified_event_date': '2012-06-17',
  'read_status': 'contemporaneous_wire_opening_read',
  'evidence_summary': 'Opening explicitly attributes church attacks and subsequent Christian youth rioting targeting Muslims to Sunday. Sunday precedes Monday June18. This corroborates an actual riot but supplies event-date counterevidence for the identified episode, not a complete exclusion of violence on June18 or proof of AFP Story16905273 identity.'},
 'NIGERIA_RIOT_VOA_JUNE18': {
  'url': 'https://www.voanews.com/a/death-toll-rises-to-50-in-nigeria-violence/1212679.html',
  'title': 'Death Toll Rises to 50 In Nigeria Violence', 'publisher': 'VOA News',
  'publication_date': '2012-06-18', 'identified_event_date': '2012-06-17',
  'read_status': 'contemporaneous_article_body_read',
  'evidence_summary': 'Describes deaths Sunday in bombings and ensuing riots; Monday is updated death toll. Monday evening gunfire in Damaturu is separately reported and its connection unresolved. Does not establish exact June18 sampled riot or identify Hungarian criminal victims.'},
 'NIGERIA_RIOT_REUTERS_JUNE18': {
  'url': 'https://www.hurriyetdailynews.com/nigeria-religious-riots-kill-at-least-52-23481',
  'title': 'Nigeria religious riots kill at least 52', 'publisher': 'Reuters via Hurriyet',
  'publication_date': '2012-06-18', 'identified_event_date': '2012-06-17',
  'read_status': 'contemporaneous_wire_article_body_read',
  'evidence_summary': 'Reuters reporter visits hospitals Monday; story explicitly dates onset of rioting to yesterday and reports Sunday curfew largely restoring order. Date mismatch indication, not a comprehensive negative claim for all June18 Nigerian violence.'},
 'NIGERIA_AFP_INQUIRER_FETCH': {
  'url': 'https://newsinfo.inquirer.net/214821/death-toll-from-nigeria-church-attacks-riots-rises-to-45',
  'read_status': 'HTTP403_no_body_read'},
 'THAI_CAMBODIA_VOA_NOV2009': {
  'url': 'https://www.voanews.com/a/a-13-2009-11-16-voa11-70423612/415739.html',
  'title': 'Thai-Cambodia Dispute Raises Nationalist Sentiment in Bangkok',
  'read_status': 'contemporaneous_article_body_read',
  'displayed_publication_date': '2009-11-18', 'url_embedded_date': '2009-11-16',
  'evidence_summary': 'Describes Sunday anti-Thaksin rally and prior cross-border dispute. URL date differs from displayed publication date; country-level exact November16 criticism cannot be confirmed from date alone. No evidence here establishes or excludes criticism of China.'},
 'THAI_CAMBODIA_WSWS_NOV16': {
  'url': 'https://www.wsws.org/en/articles/2009/11/thai-n16.pdf',
  'title': 'Thai-Cambodian tensions escalate over Thaksin appointment',
  'publication_date': '2009-11-16', 'read_status': 'contemporaneous_commentary_PDF_text_read',
  'evidence_summary': 'Discusses events last week, November5 ambassador recall, earlier ASEAN disputes and historical political tensions. Publication date is insufficient to date a new Thai criticism of Cambodia to November16. No negative truth inferred for China.'},
 'NAZOURI_SOBHI_KHALEEJ': {
  'url': 'https://www.alkhaleej.ae/2014-09-04/الأمم-المتحدة-تحذر-من-وقوع-انتهاكات-جسيمة-في-ليبيا/العالم/سياسة',
  'title': 'الأمم المتحدة تحذر من وقوع انتهاكات جسيمة في ليبيا',
  'displayed_publication_date': '2014-09-05', 'url_embedded_date': '2014-09-04',
  'read_status': 'contemporaneous_Arabic_article_body_read',
  'evidence_summary': 'Names Libyan chief Abdul Raziq al-Nazouri discussing military cooperation and border control with Sedki Sobhi in Cairo. Lead also dates separate Benghazi attacks to yesterday Thursday. Exact date of the meeting is not explicit; cannot attach yesterday to every clause or treat Friday publication as meeting date. Article also mentions earlier Hegazy meeting. Same-role meeting corroborated, sampled exact date unresolved.'},
 'LABOUR_VANGUARD_AGREEMENT_FETCH': {
  'url': 'https://www.vanguardngr.com/2011/07/fg-labour-agreement-that-aborted-the-strike/',
  'title': 'FG, Labour agreement that aborted the strike',
  'read_status': 'site_navigation_only_no_relevant_article_body_read'},
 'LABOUR_VANGUARD_CONTROVERSY_FETCH': {
  'url': 'https://www.vanguardngr.com/2011/07/fg-meets-labour-over-latest-controversy-on-n18000-new-minimum-wage/',
  'title': 'FG meets labour over latest controversy on N18,000 new minimum wage',
  'read_status': 'no_relevant_article_body_read',
  'evidence_summary': 'Web open returned page metadata but no relevant article text. Wage-dispute hypothesis remains unlinked to AllAfrica Story15699991 in Ibadan.'}
})
unknown = {'final_event_truth': 'unknown', 'original_top1_truth': 'unknown', 'gold_truth': 'unknown', 'reasonable_rank_adjustment': 'unknown'}
updates = {
 'S06': {'gold_truth': 'supported_at_query_semantic_granularity',
  'evidence_ids': ['LANG_BEIJING_AP_EXCERPT', 'LANG_TAIWAN_FETCH'],
  'rationale': 'Read dated AP excerpt supports November5 Jack Lang arrival in Beijing. Final rawTop1 France unchanged; it does not explain new rank loss. Cache identifies The Hague as sole newly overtaking candidate (+0.3540363 after, -6.7071538 before). Generic French envoy may mean other people; without comprehensive itinerary or their identities The Hague cannot be proved false from Lang being in China.'},
 'S07': {'evidence_ids': ['THAI_CAMBODIA_VOA_NOV2009','THAI_CAMBODIA_WSWS_NOV16'],
  'rationale': 'Regional dispute corroborated but exact November16 criticism not established. China is sole new filtered outranker (+0.2151604 after, -3.8191605 before); no direct support or contradiction for that same-day criticism.'},
 'S16': {'evidence_ids': ['NAZOURI_SOBHI_KHALEEJ'],
  'rationale': 'Libyan chief/Sobhi consultation corroborated in contemporary body but exact September5 event date not established. Eight filtered candidates newly overtake, all previously in candidate space; no new entity expansion in this sample. Sisi consultation is not disproved by another meeting.'},
 'S17': {'gold_truth': 'supported_at_query_semantic_granularity',
  'evidence_ids': ['LEE_MEDVEDEV_PLANNED_NOV10','LEE_JOONGANG_FETCH','LEE_HERALD_FETCH'],
  'rationale': 'Contemporary November10 report supports planned Lee/Medvedev bilateral afternoon meeting. Japan final rawTop1 is a benchmark positive but still externally unverified and already filtered. Seven new filtered outrankers are UAE/Akihito/Obama/Merkel/Mongolian head/Kazakhstan/SouthKorea, all previously present. Japan cannot explain gold filteredrank1->8. Obama/Merkel planned talks potentially genuine; not assumed false.'},
 'S28': {'evidence_ids': ['NIGERIA_RIOT_AFP_JUNE18','NIGERIA_RIOT_VOA_JUNE18','NIGERIA_RIOT_REUTERS_JUNE18','NIGERIA_AFP_INQUIRER_FETCH'],
  'rationale': 'Three read contemporary sources place identified attacks/rioting on June17 while reporting June18. Exact AFP Story16905273 and all June18 violence unlinked; retain unknown gold and final Hungarian criminal truth. 33 new outrankers and3 cease to outrank explains net30 loss; absurd-looking participants are not automatically negative evidence.'}
}
for sid,j in updates.items():
 e['judgments'][sid] = {**unknown,'stage':'detailed_source_review',**j}
e['judgments']['S40'] = {**unknown,'stage':'detailed_source_attempt_blocked',
 'evidence_ids':['LABOUR_VANGUARD_AGREEMENT_FETCH','LABOUR_VANGUARD_CONTROVERSY_FETCH'],
 'rationale':'Candidate labour-wage source bodies not acquired; exact AllAfrica Ibadan story unidentified. No semantic truth for original labour cooperation or final Obama cooperation from these reads. No replacement sample.'}
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'round5-source-notes.md').write_text('''# 第五轮历史正文核对

固定40样本保持不变，新详细核对S06/S07/S16/S17/S28，S40仅获取失败。S06的2009年11月5日法国特使Jack Lang抵京，有11月6日文章实际引用的AP开头支持；原AP链接未取得正文。S17的2010年11月10日下午李明博/梅德韦杰夫拟会谈有Daily NK正文支持，符合意向关系。原gold同日语义支持增至8条；不是新增现实错误计数。

S06最终raw第一名法国前后未变；真正新越位的是海牙。通用“法国特使”可能对应多人，不能因Jack Lang抵京就排除任何法国特使同日在海牙。S17最终raw第一名日本已被filtered排除，真实也不能解释gold1→8；7个实际新越位候选仍需逐一核对。所有这些新越位都是原候选参数变化，不是新增实体进入。

S28的6月18日AFP/Reuters/VOA报道明确把已识别教堂袭击和后续骚乱放在周日6月17日；不是新取得同日gold支持。原AFP故事身份、18日其他暴力均未完整排除。S16阿拉伯语正文支持Nazouri/Sobhi确有会谈，但未明确会谈日期；同一句的“昨天周四”明确对应班加西袭击，不能随意延伸到另一会谈。S07泰柬纠纷正文主要回述前一周，URL与页面刊发日期也不一致，不能以刊发日期确认同日批评。

当前22条详细正文核对、4条来源获取失败、1条编码身份核对、13条初检。最终Top1仍支持2/40、严格证伪0、未定38；1条条件漏标解释及S38错误成分见证保留。总体合理/错误比例未识别。没有新训练或评分；继续读既有缓存，不更改原基准。
''',encoding='utf-8')
print('ROUND5_SAVED', sorted(updates))
