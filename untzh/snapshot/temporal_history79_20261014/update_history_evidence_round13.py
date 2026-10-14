"""Append actually read source dates; do not convert missing evidence into truth."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = ROOT / 'representative-history-evidence.json'
e = json.loads(p.read_text(encoding='utf-8'))
e['sources'].update({
    'R13_CHINA_YEMEN_PROCURATORATE_REPRINT': {
        'url': 'https://qingdaonews.com/content/2009-04/20/content_8033364.htm',
        'title': '曹建明：加强中国与也门司法领域合作',
        'read_status': 'publisher_article_body_read',
        'publication_date': '2009-04-20',
        'attribution': '检察日报; author李腾',
        'identified_event_date': '2009-04-18',
        'evidence_summary': 'Body explicitly dates Cao Jianming speech and bilateral talks with Yemen procurator general to18th; other same-day meetings also described. Corroborates MFA identified-event date. Does not link internalStory10806432 or exclude all April19 China/Yemen consultations.'
    },
    'R13_KIKWETE_CONDOLENCE_REPRINT': {
        'url': 'https://www.thehabari.com/ikulu-yamtumia-rambirambi-rc-mbeya-vifo-vya-watu-10/',
        'read_status': 'article_body_read_via_remote_normal_HTTPS_after_web_tool_failure',
        'artifact': 'sources/r13_kikwete_condolence.html',
        'publication_date': '2014-09-01',
        'evidence_summary': 'Actual body attributes road-safety remarks and condolences for August29 accident to Kikwete, but contains no explicit September1 issuance date. Not sampled AllAfrica Mpwapwa story. Official counterpart later read and dates issuance August31.'
    },
    'R13_KIKWETE_CONDOLENCE_PRIMARY': {
        'url': 'https://www.ikulu.go.tz/news/press-release-7',
        'read_status': 'official_presidential_press_release_body_read',
        'displayed_page_date': '2014-07-31',
        'body_issuance_date': '2014-08-31',
        'accident_date': '2014-08-29',
        'evidence_summary': 'Presidential communications directorate signs body31Agosti2014. Page top31stJul2014 conflicts with body and later accident; preserve discrepancy. September1 reprint is not evidence of newly issued speech onSeptember1. This different source event does not disprove all broad Kikwete statements thatday.'
    },
    'R13_KIKWETE_TOUR_PRIMARY': {
        'url': 'https://www.ikulu.go.tz/documents/press-releases?page=49',
        'read_status': 'official_archive_body_read_initial_open; later_scroll_requests_timed_out',
        'identified_event_dates': ['2014-08-28', '2014-08-29', '2014-09-01', '2014-09-02'],
        'evidence_summary': 'Visible original archive body explicitly dates Gulwebridge/Mpwapwa foundation laying toAug29 and priorMbande/Kongwa activity toAug28; Sept2release dates Nairobiarrival topreviousnightSept1. Relevant tour context, not identification of AllAfricaStory35330934 or exclusion of otherSept1statements.'
    },
    'R13_CHINADAILY_LIST_NOW_NOT_FOUND': {
        'url': 'https://www.chinadaily.com.cn/cndy/2014-10/28/p3_5.html',
        'read_status': 'publisher_not_found_page_read_this_round',
        'evidence_summary': 'Currentresponseisnot-foundtemplate. Priorround openingread retained as historicalreceipt, not overwritten or newly counted as full article. IFRC China/Iran same-dayvisit unresolved.'
    },
    'R13_IFRC_MIRROR_FETCH_FAILED': {
        'url': 'https://family-care-foundation.net/the-challenges-facing-humanitarian-agencies-and-workers/',
        'read_status': 'web_fetch_failed_and_remote_SSL_UNEXPECTED_EOF_no_body',
        'evidence_summary': 'No actual op-edbody obtained; snippets not used as same-dayvisit truth.'
    }
})
def attach(sid, ids, note):
    j = e['judgments'][sid]
    j['evidence_ids'] = list(dict.fromkeys(j['evidence_ids'] + ids))
    j['round13_source_note'] = note
attach('S39', ['R13_CHINA_YEMEN_PROCURATORATE_REPRINT'],
       'Actual procuratorate newspaper reprint corroborates18April date for identifiedtalks. ChinaApril19gold stillunknown; Iraqsupportedalternative and conditionalfilterexplanation unchanged.')
attach('S27', ['R13_KIKWETE_CONDOLENCE_REPRINT','R13_KIKWETE_CONDOLENCE_PRIMARY','R13_KIKWETE_TOUR_PRIMARY'],
       'Previously inaccessible reprint body nowread; officialcorrespondingrelease signedAug31, notSept1. Sample raw has AllAfricaStory35330934, Mpwapwa, Kikwete sentence3/Magufuli sentence6. Readsources are not proven sameinternalstory. BroadgoldSept1 and soleafterTop1DomesticAffairs remainunknown; twoactualcrossings DomesticAffairs/Ministry bothunknown.')
e['judgments']['S27']['stage'] = 'detailed_source_review'
attach('S14', ['R13_CHINADAILY_LIST_NOW_NOT_FOUND','R13_IFRC_MIRROR_FETCH_FAILED'],
       'CurrentChinaDailylistingnotfound; op-edmirrorfetchfailed. Prioropeningreceipt retained; goldMakeVisitChinaOct25 and solecrossingIran notadjudicated.')
out = {
    'identified_source_events': [
        {'sample_id':'S39','source_id':'R13_CHINA_YEMEN_PROCURATORATE_REPRINT','identified_event_date':'2009-04-18','gold_query_date':'2009-04-19','same_internal_story_proven':False},
        {'sample_id':'S27','source_id':'R13_KIKWETE_CONDOLENCE_PRIMARY','identified_event_date':'2014-08-31','gold_query_date':'2014-09-01','same_internal_story_proven':False}
    ],
    'new_gold_support_n':0,'new_final_top1_support_n':0,'new_whole_error_drift_n':0,
    'world_reasonable_fraction':None,'world_error_fraction':None,
    'training':False,'neural_rescoring':False,
    'note':'Different source-event dates cannot exhaustively disprove broad daily query; source availability is not sample replacement criterion.'
}
(ROOT/'identified-source-event-dates-round13.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert sum(j['gold_truth']=='supported_at_query_semantic_granularity' for j in e['judgments'].values())==9
assert sum(j['final_event_truth']=='supported_at_query_semantic_granularity' for j in e['judgments'].values())==2
print(json.dumps(out,ensure_ascii=False))
