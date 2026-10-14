"""Record independently read histories and cached score crossings, preserving unknowns."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

e = read('representative-history-evidence.json')
sources = {
 'AAP_FORMATION_OFFICIAL': {
  'url': 'https://archive.aamaadmiparty.org/whatever-results-but-name-of-aap-will-be-written-in-history-for-fighting-elections-with-so-much-courage-in-these-circumstances-atishi/',
  'title': 'AAP post-election office-bearer meeting, June 13 2024',
  'read_status': 'official_article_body_read', 'publication_date': '2024-06-13',
  'organization_formation_date': '2012-11-26',
  'evidence_summary': 'Official party release attributes the November 26 2012 formation date to Sanjay Singh. Aam Aadmi Party is the Indian AAP, not BJP.'},
 'LNP_FORMATION_OFFICIAL': {
  'url': 'https://www.lnp.org.au/about', 'title': 'About the LNP',
  'read_status': 'official_party_history_body_read', 'organization_formation_date': '2008-07-26',
  'evidence_summary': 'Queensland Liberal and National branches voted to form the LNP on July 26 2008. Earlier predecessor parties existed; they are not the same named organization.'},
 'LNP_AEC_REGISTRATION': {
  'url': 'https://www.aec.gov.au/parties_and_representatives/party_registration/Registration_Decisions/2009/lnp.htm',
  'title': 'Party registration decision: Liberal National Party of Queensland',
  'read_status': 'official_registration_decision_body_read',
  'evidence_summary': 'AEC received an application August 22 2008, and recorded its decision February 18 2009. Registration is distinct from formation. Mentions a constitution dated October 6 2007; this too postdates the 2005 query.'},
 'FREE_EGYPTIANS_LAUNCH': {
  'url': 'https://www.egyptindependent.com/naguib-sawiris-launches-liberal-free-egyptians-party/',
  'title': "Naguib Sawiris launches liberal Free Egyptians Party",
  'read_status': 'contemporaneous_article_body_read', 'publication_date': '2011-04-04',
  'organization_launch_date': '2011-04-03',
  'evidence_summary': 'Monday April 4 report describes a Sunday press-conference launch of the new party. Launch date is not registration date; supports named organization postdating 2005.'},
 'M23_CANADA_HISTORY': {
  'url': 'https://www.irb-cisr.gc.ca/en/country-information/rir/Pages/index.aspx?doc=454490&pls=1',
  'title': 'DRC: M23, activities and affiliations (2012-January 2013)',
  'read_status': 'government_country_report_body_read', 'publication_date': '2013-01-30',
  'organization_formation_month': '2012-04',
  'evidence_summary': 'Report describes M23 arising from the April 2012 army mutiny. Name refers to a March 23 2009 agreement. Former CNDP members and earlier militias are predecessors, not proof of M23 existence in 2005.'},
 'NUSRA_USDOS_HISTORY': {
  'url': 'https://www.refworld.org/reference/annualreport/usdos/2015/105472',
  'title': 'Country Reports on Terrorism 2014: al-Nusrah Front',
  'read_status': 'US_State_Department_report_body_read_on_Refworld', 'publication_date': '2015-06-19',
  'organization_formation_period': 'late 2011',
  'evidence_summary': 'Report explicitly dates formation to late 2011 and separation from AQI to 2013. This supplies formation evidence; public emergence alone would not exclude secret earlier existence.'},
 'NUSRA_UN_HISTORY': {
  'url': 'https://main.un.org/securitycouncil/en/sanctions/1267/aq_sanctions_list/summaries/entity/al-qaida-in-iraq',
  'title': 'Security Council narrative summary: Al-Qaida in Iraq',
  'read_status': 'official_UN_narrative_body_read', 'public_emergence_date': '2012-01-24',
  'evidence_summary': 'UN dates Jabhat al Nusrah public emergence to January 24 2012 and explains its creation by AQI members. Distinguish organizational identity from earlier AQI and from the same individuals before membership.'},
 'CNS_LIBRARY_OF_CONGRESS': {
  'url': 'https://tile.loc.gov/storage-services/master/frd/copr/Thailand.pdf',
  'title': 'Library of Congress Country Profile: Thailand, July 2007',
  'read_status': 'government_report_relevant_pdf_text_read', 'publication_date': '2007-07',
  'evidence_summary': 'Pages 18-20 relate the CNS military junta to the September 2006 coup. This does not identify every ICEWS use of Council for National Security as that junta.',
  'identity_caution': 'Raw ICEWS records use the same name for Thai governmental security entities January 1 and March 6 2006, before the coup. Possible conflation with an older National Security Council; exclude from confirmed formation witnesses.'},
 'WINDSOR_ABC_FEB3': {
  'url': 'https://www.abc.net.au/worldtoday/archive/articles/2005/02/03/s1295318.htm',
  'title': 'Tony Windsor stands by bribery claims to Senate inquiry',
  'read_status': 'ABC_transcript_body_read', 'publication_date': '2005-02-03',
  'evidence_summary': 'Windsor describes businessman Greg Maguire allegedly conveying an inducement in May 2004; others deny claims. Not evidence of a new policy-support event February 4 2005.'},
 'WINDSOR_ABC_MAR11': {
  'url': 'https://www.abc.net.au/am/archive/articles/2005/03/11/s1321056.htm',
  'title': 'Maquire explains bribery claims to Senate inquiry',
  'read_status': 'ABC_transcript_body_read', 'publication_date': '2005-03-11',
  'evidence_summary': 'Maguire describes prior dealings and says he formerly supported and helped fund Windsor. Does not establish February 4 policy support, and disputed allegations are not adjudicated facts.'},
 'WINDSOR_PARLIAMENT_FETCH': {
  'url': 'https://www.aph.gov.au/Parliamentary_Business/Committees/Senate/Finance_and_Public_Administration/Completed_inquiries/2004-07/reg_partner_prog/report/c08',
  'title': 'Senate regional partnerships inquiry: chapter 8',
  'read_status': 'fetch_restricted_no_body_read', 'evidence_summary': 'Primary report fetch failed; no substantive label from failure.'},
 'SLOVAK_HUNGARY_SITA_SEP17': {
  'url': 'https://www.pravda.sk/spravy/domace/clanok/163420-lajcak-sa-hneva-madarsky-velvyslanec-musel-ist-na-kobercek',
  'title': 'Lajčák sa hnevá. Maďarský veľvyslanec musel ísť na koberček',
  'publisher': 'SITA/CTK through Pravda',
  'read_status': 'contemporaneous_wire_article_body_read', 'publication_date': '2009-09-17', 'event_date': '2009-09-17',
  'evidence_summary': 'Same-day wire report records Slovak foreign ministry rejecting Hungarian minister Balazs remarks and summoning the Hungarian ambassador that afternoon. It attributes the ministry response to spokesperson Peter Stano.',
  'mapping': 'Supports Foreign Affairs (Slovakia) / Criticize or denounce / Foreign Affairs (Hungary) at ministerial institutional granularity. Exact ICEWS Story ID linkage remains unproven.'}
}
e['sources'].update(sources)
unknown = {'stage': 'detailed_source_review', 'final_event_truth': 'unknown',
           'original_top1_truth': 'unknown', 'gold_truth': 'unknown',
           'reasonable_rank_adjustment': 'unknown'}
e['judgments']['S21'] = {**unknown,
 'evidence_ids': ['LNP_FORMATION_OFFICIAL', 'LNP_AEC_REGISTRATION', 'WINDSOR_ABC_FEB3', 'WINDSOR_ABC_MAR11', 'WINDSOR_PARLIAMENT_FETCH'],
 'rationale': 'LNP is a newly outranking candidate, not raw Top1. Australia identity supported by raw country metadata; predecessor/merged-party naming remains an explicit condition. Gold exact dated event and final citizen support remain unknown. Read February and March Senate reports concern earlier events; no label based on plausibility or missing mention.'}
e['judgments']['S22'] = {**unknown,
 'evidence_ids': ['CNS_LIBRARY_OF_CONGRESS'],
 'rationale': 'CNS may refer to a junta or a preexisting Thai security institution. Raw January/March 2006 records predate the coup. Do not use the junta formation date as a strict negative for the ambiguous entity. IDF self-consultation and final Hanegbi consultation remain unverified.'}
e['judgments']['S38'] = {**unknown, 'gold_truth': 'supported_at_query_semantic_granularity',
 'evidence_ids': ['SLOVAK_HUNGARY_SITA_SEP17', 'NUSRA_USDOS_HISTORY', 'NUSRA_UN_HISTORY'],
 'rationale': 'Same-day ministerial criticism supports gold. A newly introduced, unfiltered Terrorist (Jabhat al-Nusra) candidate outranks it after learning. Under dated named-group-affiliation semantics, late-2011 formation excludes such affiliation in 2009. This demonstrates one erroneous rank component; not all 158 lost positions, not raw Top1 truth, not full-query exclusive error classification.'}
for sid in ['S04']:
    e['judgments'][sid]['evidence_ids'] = list(dict.fromkeys(e['judgments'][sid]['evidence_ids'] +
       ['AAP_FORMATION_OFFICIAL', 'FREE_EGYPTIANS_LAUNCH', 'M23_CANADA_HISTORY', 'NUSRA_USDOS_HISTORY']))
save('representative-history-evidence.json', e)

crossings = read('fixed40-new-outrankers.json')
index = {r['sample_id']: r for r in crossings['records']}
definitions = [
 ('S04', 'Aam Aadmi Party', ['AAP_FORMATION_OFFICIAL']),
 ('S04', 'Free Egyptians Party', ['FREE_EGYPTIANS_LAUNCH']),
 ('S04', 'March 23 Movement', ['M23_CANADA_HISTORY']),
 ('S04', 'Terrorist (Jabhat al-Nusra)', ['NUSRA_USDOS_HISTORY', 'NUSRA_UN_HISTORY']),
 ('S04', 'Combatant (Jabhat al-Nusra)', ['NUSRA_USDOS_HISTORY', 'NUSRA_UN_HISTORY']),
 ('S21', 'Liberal National Party', ['LNP_FORMATION_OFFICIAL', 'LNP_AEC_REGISTRATION']),
 ('S38', 'Terrorist (Jabhat al-Nusra)', ['NUSRA_USDOS_HISTORY', 'NUSRA_UN_HISTORY'])
]
witnesses = []
for sid, name, source_ids in definitions:
    rec = index[sid]
    cs = [c for c in rec['new_outrankers'] if name in c['fact']]
    assert len(cs) == 1
    cand = cs[0]
    assert cand['after_gap'] > 0 and cand['was_new_candidate']
    witnesses.append({'sample_id': sid, 'candidate_name': name, **cand,
      'gold_fact': rec['gold_fact'], 'original_rank': rec['original_rank'], 'final_rank': rec['final_rank'],
      'source_ids': source_ids,
      'gold_independent_support': e['judgments'][sid]['gold_truth'] == 'supported_at_query_semantic_granularity',
      'label': 'temporally_incompatible_under_named_organization_or_dated_affiliation_semantics',
      'counterfactual_rank_reduction_if_this_candidate_removed': 1,
      'qualification': 'Not a raw Top1 label. Group existence is distinct from earlier members/predecessors. TRAIN first occurrence is not real formation date. Removal is a score-list diagnostic, not retraining or proof of complete rank justification.'})
query_ids = sorted({w['sample_id'] for w in witnesses})
supported_ids = sorted({w['sample_id'] for w in witnesses if w['gold_independent_support']})
out = {
 'fixed_sample_n': 40, 'population_worsened_n': 42851,
 'new_strict_outranker_total_n': sum(r['newly_strictly_above_gold_n'] for r in crossings['records']),
 'temporal_witness_candidate_n': len(witnesses), 'temporal_witness_query_ids': query_ids,
 'temporal_witness_query_n': len(query_ids), 'temporal_witness_query_coverage': len(query_ids) / 40,
 'supported_gold_with_temporal_error_component_query_ids': supported_ids,
 'supported_gold_with_temporal_error_component_query_n': len(supported_ids),
 'supported_gold_with_temporal_error_component_sample_lower_bound_conditional': len(supported_ids) / 40,
 'reality_error_population_fraction': None, 'reasonable_whole_rank_fraction': None,
 'conditions': 'Organization/affiliation names refer to the named organization at query date, rather than retroactive aliases for predecessors. Source mapping is at benchmark semantic granularity. Fixed-sample witness coverage is not a population estimate or complete UFR.',
 'excluded_identity_ambiguous_witnesses': [{'sample_ids': ['S04', 'S22'], 'candidate': 'Council for National Security',
   'reason': 'Country Thailand confirmed, but raw dates January/March 2006 predate coup junta; same name may cover longstanding security institution. Formation identity not resolved.'}],
 'costs': {'filtered_top5_diagnostic_scoring_seconds': read('fixed40-filtered-competitors.json')['seconds'],
           'all_new_outranker_diagnostic_scoring_seconds': crossings['seconds'],
           'raw_identity_context_seconds': read('competitor-identity-context.json')['seconds'],
           'scope': 'These two diagnostics did perform new neural scoring of unchanged checkpoints. No training. Does not include browsing, code creation, source interpretation, prior full raw scan, or mirror.'},
 'witnesses': witnesses
}
save('temporal-competitor-evidence.json', out)
lines = ['# 固定历史查询中的错误排名成分', '',
 '仍使用原固定40条概率样本，未换样。查证对象是实际新增且未过滤、严格超过gold的竞争者；最终第一名真假另行记录。', '',
 f'全部新超过gold的候选共{out["new_strict_outranker_total_n"]}个，目前取得组织成立时间证据的候选7个，涉及3条查询（S04/S21/S38）。在接受名称对应该组织或当时成员类别的语义条件下，这3条含时间不相容的排名竞争者。3/40=7.5%是当前样本见证覆盖率，不是总体错误率。', '',
 'S38原gold另有同日通讯正文支持，因此1/40=2.5%具有“有证据支持的历史答案被时间不成立的新竞争者超过”的错误成分。它只是固定样本的条件下限，不是整体UFR点估计，也不证明剩余39条合理。S04/S21原gold精确日期未定。', '',
 '| 查询 | 日期 | 新超过gold的候选 | after分差 | 原gold独立同日支持 |', '|---|---|---|---:|---|']
for w in witnesses:
    lines.append(f'| {w["sample_id"]} | {w["gold_fact"][3]} | {w["candidate_name"]} | {w["after_gap"]:.6f} | {"有" if w["gold_independent_support"] else "未定"} |')
lines += ['',
 'S38：2009-09-17斯洛伐克外交部批评匈牙利外交部，rank4→162。[同日SITA/ČTK正文](https://www.pravda.sk/spravy/domace/clanok/163420-lajcak-sa-hneva-madarsky-velvyslanec-musel-ist-na-kobercek)支持原gold；[美国国务院报告](https://www.refworld.org/reference/annualreport/usdos/2015/105472)将Jabhat al-Nusra成立时间限定在2011年末，联合国另记录2012年1月公开出现。其成员类别在2009年不成立。竞争者分数比gold高0.097689，新增并且未过滤；移除这一个候选会使after rank162降为161。其余157个净新增位置仍未判定，不能把158个位置全算错误。', '',
 'S04：2005年查询出现2012年成立的AAP/M23和2011年的Free Egyptians/Nusra类别。原Lucy抗议日期已有回述疑点，因此不把这些错误候选自动当成已验证真gold的遗忘。', '',
 'S21：2005年Windsor查询中LNP进入gold上方；官方党史写2008年7月成立、AEC注册决定为2009年2月。raw国家为Australia，但2008年5月已有同名编码，可能报道拟组党或名称归并。2005查询的组织身份条件必须显式保留。原gold仍缺精确同日支持，ABC参院报道不能用过去的资助或争议诱因补成当日支持。', '',
 '排除CNS见证：美国国会图书馆报告说明2006年政变军政府，但raw同名泰国安全机构在2006年1月和3月已出现。不能未经消歧就将军政府成立日套到所有同名编码。S22保留未知。', '',
 '此前两次诊断重新评分耗时0.943秒、1.031秒，已缓存40条全部before/after分数；本次只读这些既有结果，新做身份上下文读取8.811秒。没有训练或本次新评分。早期来源核对“无新评分”的陈述仅适用于那个阶段，两次实际评分成本仍保留。', '',
 '合理比例和真正错误比例仍未完整识别。1条S39有条件漏标解释、1条S38有错误排名成分，两者不能组成穷尽二分类；混合原因和证据不足仍需保留。']
(ROOT / '错误排名成分核对.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'witnesses'}, ensure_ascii=False, indent=2))
