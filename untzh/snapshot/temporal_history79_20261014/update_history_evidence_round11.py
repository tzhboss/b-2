"""Append new actual source reads while retaining unresolved query-level truth."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = ROOT / 'representative-history-evidence.json'
e = json.loads(p.read_text(encoding='utf-8'))
e['sources'].update({
    'R11_KIBAKI_MAY16_IOL': {
        'url': 'https://iol.co.za/news/africa/2005-05-16-state-intervenes-in-kibakis-legal-wrangle/',
        'title': "State intervenes in Kibaki's legal wrangle",
        'read_status': 'contemporaneous_article_body_read',
        'publication_date_in_url': '2005-05-16',
        'identified_incident_date': '2005-05-02',
        'evidence_summary': 'Same-day legal-action report explicitly separates Monday lawsuit from newsroom incident late May2. Corroborates earlier judgment mirror. Page body does not display AFP attribution or link internal ICEWS StoryID; do not assert exact original article identity or exclude every May16 incident.'
    },
    'R11_KIBAKI_RSF': {
        'url': 'https://rsf.org/en/assault-journalists-first-ladies-eve-world-press-freedom-day',
        'read_status': 'organization_original_report_body_read',
        'publication_date': '2005-05-04',
        'identified_incident_start_date': '2005-05-02',
        'evidence_summary': 'RSF describes Lucy arriving at Nation Media Group shortly before midnight May2, with five-hour occupation and cameraman assault. Independent corroboration of identified incident date, not exhaustive absence evidence for May16.'
    },
    'R11_KENYALAW_SOURCE_FETCH': {
        'url': 'https://new.kenyalaw.org/akn/ke/judgment/kehc/2006/3506/eng%402006-11-17/source',
        'read_status': 'web_fetch_failed_403_no_primary_body',
        'evidence_summary': 'Different source resource attempted; search preview not substituted for primary court body.'
    },
    'R11_WINDSOR_PRIMARY_FETCH': {
        'urls': [
            'https://www.aph.gov.au/Parliamentary_Business/Committees/Senate/Finance_and_Public_Administration/Completed_inquiries/2004-07/reg_partner_prog/report/c01',
            'https://www.aph.gov.au/binaries/hansard/senate/commttee/s8053.pdf',
            'https://www.aph.gov.au/Parliamentary_Business/Committees/Senate/Privileges/Completed_inquiries/2005-2009/report_131/c01'
        ],
        'read_status': 'web_open_failed_no_primary_body',
        'evidence_summary': 'Indexed contributions testimony is not independently read Feb4 intent evidence; no truth labels from previews.'
    },
    'R11_CHEN_MAMI_BODY': {
        'url': 'https://news.cntv.cn/china/20120110/104227.shtml',
        'title': '胡锦涛主席特使陈至立会见哈萨克斯坦议会上院议长',
        'read_status': 'CNTV_PeoplesDaily_Xinhua_republication_body_read',
        'publication_date': '2012-01-10',
        'publication_time': '05:18',
        'article_dateline': 'Almaty2012-01-09',
        'identified_meeting_date': '2012-01-09',
        'evidence_summary': 'Body explicitly dates Chen/Mami Astana consultation to Jan9 and describes parliamentary/cooperation discussion. Earlier Mami evidence was preview only; this is an actual new body read. Broad Kazakhstan-legislature actor category, exact BBCMonitoring StoryID, and separate Jan10 consultation remain unresolved.'
    },
    'R11_CHEN_PRIME_MINISTER': {
        'url': 'https://news.cntv.cn/program/xwlb/20120110/123375.shtml',
        'read_status': 'CNTV_original_news_body_read',
        'publication_date': '2012-01-10',
        'identified_meeting_date': '2012-01-09',
        'evidence_summary': 'Dates meeting with Kazakh prime minister Masimov to Jan9. Different office from gold legislature; used as itinerary context only.'
    },
    'R11_CHEN_CHINANEWS_FETCH': {
        'url': 'https://www.chinanews.com/gn/2012/01-09/3591628.shtml',
        'read_status': 'web_not_accessible_no_body_read',
        'evidence_summary': 'Preview not used for labels; subsequently followed actual CNTV article link.'
    },
    'R11_ONDO_LABOUR': {
        'url': 'https://www.vanguardngr.com/2011/07/ondo-workers-suspend-strike/',
        'read_status': 'contemporaneous_original_news_body_read',
        'publication_date': '2011-07-26',
        'identified_event_date': '2011-07-25',
        'evidence_summary': 'Akure/Ondo strike ending yesterday; different location and date from sampled Ibadan fact. Excluded as direct same-day support.'
    }
})

def append(sid, ids, text):
    j = e['judgments'][sid]
    j['evidence_ids'] = list(dict.fromkeys(j['evidence_ids'] + ids))
    j['rationale'] = j['rationale'].split(' Round11:')[0] + ' Round11: ' + text

append('S04', ['R11_KIBAKI_MAY16_IOL', 'R11_KIBAKI_RSF', 'R11_KENYALAW_SOURCE_FETCH'],
       'Actual May16 lawsuit body and RSF original report independently place identified Nation incident lateMay2/earlyMay3. Original rawMay16story sentence2lawsuit/sentence5riot is consistent with retrospective-date contamination. No exact AFPStoryID linkage, AFP attribution on new IOLbody, or exhaustive May16 exclusion. Keep gold and Raheel truths unresolved; cannot count erroneous drift against unsupported gold.')
append('S21', ['R11_WINDSOR_PRIMARY_FETCH'],
       'New primary privileges-report route also unavailable; indexed historical campaign contributions do not establish Feb4 new intent. Preserve unknown truth and existing LNP conditional negative; no new strict error label.')
append('S31', ['R11_CHEN_MAMI_BODY', 'R11_CHEN_PRIME_MINISTER', 'R11_CHEN_CHINANEWS_FETCH'],
       'Actual Jan10CNTV/PeoplesDaily/Xinhua body explicitly places Mami meeting Jan9. This strengthens identified-event date counterevidence beyond earlier preview. Gold actor remains broad legislature, not specifically Mami; exact BBCMonitoringStoryID and separate Jan10 consultation unproven. Do not label whole gold false or new Shura prediction false.')
append('S40', ['R11_ONDO_LABOUR'],
       'Newly acquired original July26Vanguard body concerns Akure/Ondo yesterday, not sampled Ibadan same-day cooperation; excluded. Existing labour articles remain unacquired. No gold/final truth label.')
e['judgments']['S40']['stage'] = 'detailed_source_review'

diagnostic = {
    'sample_n': 40,
    'sample_replaced': False,
    'scope': 'Date counterevidence for identified event counterparts, not query-level false labels or population date-error estimates.',
    'records': [
        {'sample_id': 'S04', 'benchmark_date': '2005-05-16', 'identified_event_dates': ['2005-05-02', '2005-05-03'],
         'source_ids': ['R11_KIBAKI_MAY16_IOL', 'R11_KIBAKI_RSF'],
         'identified_counterpart_date_differs': True, 'exact_original_story_link': False,
         'all_possible_query_events_excluded': False, 'gold_query_false': None},
        {'sample_id': 'S31', 'benchmark_date': '2012-01-10', 'identified_event_dates': ['2012-01-09'],
         'source_ids': ['R11_CHEN_MAMI_BODY'],
         'identified_counterpart_date_differs': True, 'exact_original_story_link': False,
         'all_possible_query_events_excluded': False, 'gold_query_false': None}
    ],
    'world_reasonable_fraction': None, 'world_error_fraction': None,
    'training': False, 'neural_rescoring': False,
    'costs': {'browsing_and_interpretation_seconds': None,
              'note': 'Not measured as one continuous stage; unknown is not zero.'}
}
(ROOT / 'identified-source-event-dates-round11.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
p.write_text(json.dumps(e, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
assert sum(j.get('gold_truth') == 'supported_at_query_semantic_granularity' for j in e['judgments'].values()) == 9
assert e['judgments']['S38']['unjustified_rank_drift'].startswith('confirmed_conditional')
print(json.dumps({'new_date_counterpart_body_ids': ['S04', 'S31'], 'gold_support_n': 9, 'world_error_fraction': None}, ensure_ascii=False))
