"""Append read official sources, preserving unresolved query truth."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = ROOT / 'representative-history-evidence.json'
e = json.loads(p.read_text(encoding='utf-8'))
e['sources'].update({
    'R14_WEN_SPECIAL_REPRESENTATIVES_PRIMARY': {
        'url': 'https://www.fmprc.gov.cn/zyxw/201005/t20100525_308073.shtml',
        'title': '温家宝会见美国总统特别代表国务卿克林顿和财长盖特纳',
        'read_status': 'official_MFA_article_body_read',
        'publication_date': '2010-05-25',
        'identified_event_date': '2010-05-25',
        'evidence_summary': 'Actual body names Clinton and Geithner as visitors and Obama special representatives in Wen meeting in Beijing May25. Not an Obama personal meeting. Internal BBC Story8255477 not linked; no exhaustive exclusion of May26 phone consultation.'
    },
    'R14_MFA_DIALOGUE_BRIEFING_PRIMARY': {
        'url': 'https://www.mfa.gov.cn/web/zyxw/201005/t20100526_308101.shtml',
        'title': '关于第二轮中美战略与经济对话的书面吹风稿',
        'read_status': 'official_MFA_briefing_body_read',
        'publication_date': '2010-05-26',
        'identified_event_dates': ['2010-05-24', '2010-05-25'],
        'evidence_summary': 'May26 publication describes May24-25 Beijing dialogue led by Clinton/Geithner as Obama representatives. Obama sent written address, Hu/Wen met US delegation leaders. Participant/date mapping counterindication, not proof about exact sampled BBC report.'
    },
    'R14_OBAMA_FREMONT_PRIMARY': {
        'url': 'https://obamawhitehouse.archives.gov/the-press-office/remarks-president-economy-0',
        'title': 'Remarks by the President on the Economy',
        'read_status': 'official_White_House_transcript_body_read',
        'publication_date': '2010-05-26',
        'speech_local_datetime': '2010-05-26T10:07:00-07:00',
        'speech_beijing_datetime': '2010-05-27T01:07:00+08:00',
        'location': 'Solyndra, Fremont, California',
        'evidence_summary': 'Read speech header and personal physical-presence remarks. This single timestamp is May27 in Beijing; not sufficient to exclude a May26 Beijing visit over the entire local day and does not exclude remote consultation. No whole-day itinerary claim.'
    },
    'R14_STATE_NEGOTIATION_TRANSCRIPT_REPRINT': {
        'url': 'https://www.iranwatch.org/library/government/united-states/executive-branch/department-state/us-dos-press-briefing-mccormack-negotiations-4-20-06',
        'title': 'Briefing with Spokesperson Sean McCormack on Negotiations Timeline (Excerpts)',
        'read_status': 'contemporary_US_State_Department_transcript_excerpt_reprint_body_read',
        'identified_event_date': '2006-04-20',
        'evidence_summary': 'US spokesman discusses possible sanctions and future Chapter7 resolution; expects Council discussion after April28 IAEA report, explicitly does not exclude daily Council discussion in interim. US spokesman statements are not automatically same-day UNSC threats. Original LeMondeStory34088749 not linked.'
    },
    'R14_IRAN_UN_LETTER_BODY_UNAVAILABLE': {
        'url': 'https://digitallibrary.un.org/record/574092/files/A_60_834--S_2006_273-EN.pdf',
        'read_status': 'web_open_failed403_remoteHTTP202_empty_body',
        'evidence_summary': 'Search preview names Rice onApril20, but actual PDF not obtained. Remote request returned202 text/html zero bytes; no source-body truth assigned. Letter circulation symbol does not turn attributed US statements into UNSC statements.'
    }
})
for sid, ids, note in [
    ('S35', ['R14_WEN_SPECIAL_REPRESENTATIVES_PRIMARY', 'R14_MFA_DIALOGUE_BRIEFING_PRIMARY', 'R14_OBAMA_FREMONT_PRIMARY'],
     'RawStory8255477 sentence1 has ObamaConsultWen, ObamaMakeVisitChina and reciprocal HostVisit/Consult onMay26. Official related event has representatives meetingMay25, withMay26 briefing. Suggests possible participant/date mapping issue, but exact BBC body unavailable. Single California timestamp isMay27Beijing; no full-day exclusion or goldfalse label. NetanyahuTop1 unchanged and stillunknown.'),
    ('S18', ['R14_STATE_NEGOTIATION_TRANSCRIPT_REPRINT', 'R14_IRAN_UN_LETTER_BODY_UNAVAILABLE'],
     'Actually read contemporary State briefing excerpt; speakerUS, future Council steps, expressly allows interim daily discussion. Does not independently establish or disprove UNSCThreatenSanctionsIran onApril20. LeMonde exactstory unlinked, gold/finalObama remainunknown.')
]:
    j=e['judgments'][sid]
    j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+ids))
    j['round14_source_note']=note
    j['stage']='detailed_source_review'
e['round14_identified_event_artifact']='identified-source-event-dates-round14.json'
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
out={
    'sample_n':40,'sample_replaced':False,
    'identified_source_events':[
        {'sample_id':'S35','source_id':'R14_WEN_SPECIAL_REPRESENTATIVES_PRIMARY','event_date':'2010-05-25','gold_query_date':'2010-05-26','actual_meeting_participants':['Wen Jiabao','Hillary Clinton','Timothy Geithner'],'Obama_role':'represented_principal','same_internal_story_proven':False},
        {'sample_id':'S18','source_id':'R14_STATE_NEGOTIATION_TRANSCRIPT_REPRINT','event_date':'2006-04-20','speaker':'US_State_Department_spokesman','gold_actor':'UN Security Council','speaker_equals_gold_actor':False,'same_internal_story_proven':False}
    ],
    'new_gold_support_n':0,'new_final_top1_support_n':0,'new_whole_error_drift_n':0,
    'world_reasonable_fraction':None,'world_error_fraction':None,
    'training':False,'neural_rescoring':False,
    'note':'Source event date, speaker identity and query truth must be kept separate. Existing nine supported golds and S38 bound retained.'
}
(ROOT/'identified-source-event-dates-round14.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert sum(j['gold_truth']=='supported_at_query_semantic_granularity' for j in e['judgments'].values())==9
assert sum(j['final_event_truth']=='supported_at_query_semantic_granularity' for j in e['judgments'].values())==2
print(json.dumps(out,ensure_ascii=False))
