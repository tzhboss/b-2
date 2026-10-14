"""Keep exact occurrence dates separate from report dates and query labels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'representative-history-evidence.json'
e = json.loads(path.read_text(encoding='utf8'))
labels_before = {k: (j['gold_truth'], j['final_event_truth'], j['reasonable_rank_adjustment'])
                 for k, j in e['judgments'].items()}
e['sources'].update({
    'R16_BOSWORTH_AP_BEIJING': {
        'url': 'https://www.cbsnews.com/news/us-urges-patience-in-n-korea-talks/',
        'read_status': 'actual_AP_body_via_CBS_read',
        'publication_date': '2009-12-12', 'publication_time_zone': 'EST',
        'source_summary': 'Bosworth spoke in Beijing Saturday; reported Chinese meetings lack explicit occurrence dates.',
        'exact_original_story_linkage': False,
    },
    'R16_BOSWORTH_PHOENIX_ORIGINAL': {
        'url': 'https://phtv.ifeng.com/program/zbjsj/200912/1212_6349_1472239.shtml',
        'read_status': 'actual_original_broadcast_text_read',
        'web_publication_date': '2009-12-12', 'broadcast_date': '2009-12-11',
        'identified_arrival_and_meeting_date': '2009-12-11',
        'evidence_summary': 'The December11 broadcast describes Bosworth arriving Beijing Friday and meeting YangJiechi/WuDawei. This dates the identified meetings to December11, while the webpage is dated December12. It does not exclude another generic US envoy consultation on December12 or establish the precise BBCMonitoringStory11283975 identity.',
    },
    'R16_BOSWORTH_STATE_INDEX_UNAVAILABLE': {
        'urls': ['https://2009-2017.state.gov/p/eap/rls/rm/2009/index.htm',
                 'https://2009-2017.state.gov/r/pa/ei/speeches/2009/index.htm'],
        'read_status': 'actual_index_fetch_failed; search_preview_only',
        'evidence_summary': 'Search results index a December12 Beijing morning walkthrough; the actual transcript was not retrieved and no meeting truth is inferred from an index preview.',
    },
    'R16_IFRC_IRNA_TEHRAN_DEC17': {
        'url': 'https://oananews.org/content/news/general/veep-iran-ready-cooperation-humanitarian-fields',
        'read_status': 'actual_IRNA_body_via_OANA_read',
        'publication_date': '2014-12-18', 'identified_meeting_date': '2014-12-17',
        'identified_date_derivation': 'ThursdayDecember18dateline; body specifies Wednesday evening meeting',
        'evidence_summary': 'IRNA reports Jahangiri meeting IFRCSecretaryGeneralElhadjAsSy in Tehran Wednesday evening. Supports December17 presence, not the sampled October25 visit; another representative or separate October25 event is not excluded.',
    },
    'R16_IFRC_OFFICIAL_2014_REPORT': {
        'url': 'https://www.ifrc.org/sites/default/files/Annual-report-2014.pdf',
        'read_status': 'actual_official_report_text_read',
        'publication_year': 2015, 'event_month': '2014-12', 'pdf_page_one_based': 12,
        'evidence_summary': 'December highlights record an Iran cooperation MOU involving SecretaryGeneralElhadjAsSy. A monthly highlight list is not a complete itinerary, and cannot exclude an October25 Iran visit or prove October25 China presence.',
    },
})
for sample_id, ids, note in [
    ('S13', ['R16_BOSWORTH_AP_BEIJING', 'R16_BOSWORTH_PHOENIX_ORIGINAL',
             'R16_BOSWORTH_STATE_INDEX_UNAVAILABLE'],
     'Actual sources supply plausible Bosworth context but date identified Chinese meetings December11. Original BBCMonitoringStory11283975sentence4 codes both Beijing consultation and retrospective Pyongyang visits onDecember12; coded rows are not source body or exact event identity. GenericEnvoyUSA does not uniquely identify Bosworth or Gration. GoldDecember12China and finalSudan consultation remain unknown.'),
    ('S14', ['R16_IFRC_IRNA_TEHRAN_DEC17', 'R16_IFRC_OFFICIAL_2014_REPORT'],
     'Actual IRNA datedbody supports December17Tehran meeting; official annualreport places related MOU inDecember. These are other-date events, not positive or negative October25Iran labels. PriorBeijingconferenceOct22-24 alone cannot certify exactOct25gold; genericorganization allows multiple representatives.'),
]:
    j = e['judgments'][sample_id]
    j['evidence_ids'] = list(dict.fromkeys(j['evidence_ids'] + ids))
    j['round16_source_note'] = note
    if sample_id == 'S13':
        j['stage'] = 'detailed_source_review'
        j['rationale'] += ' Round16: ActualAP/Phoenix sources retrieved; identified Bosworth meetings dateDecember11 rather than webpublicationDecember12. Originalstoryidentity and both queried truths stillunknown; priorGrationidentity was not established.'
e['round16_identified_event_dates_artifact'] = 'identified-event-dates-round16.json'
assert labels_before == {k: (j['gold_truth'], j['final_event_truth'], j['reasonable_rank_adjustment'])
                         for k, j in e['judgments'].items()}
assert len(e['judgments']) == 40
path.write_text(json.dumps(e, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({'sample_n':40, 'sample_replaced':False, 'truth_labels_changed':0,
                  'new_detailed_review':'S13', 'world_reasonable_fraction':None,
                  'world_error_fraction':None, 'training':False, 'neural_rescoring':False}))
