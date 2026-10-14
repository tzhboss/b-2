"""Persist actual Peru primary sources; do not infer dated actor truth from travel context."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=ROOT/'representative-history-evidence.json'
e=json.loads(p.read_text(encoding='utf8'))
before={k:tuple(j[x] for x in ('gold_truth','final_event_truth','reasonable_rank_adjustment')) for k,j in e['judgments'].items()}
e['sources'].update({
 'R17_PERU_TRAVEL_RESOLUTION_30257':{
  'url':'https://www.mef.gob.pe/contenidos/servicios_web/conectamef_quechua/pdf/normas_legales_2012/NL20141029.pdf',
  'read_status':'actual_official_gazette_body_read_page4_lines347_393',
  'publication_date':'2014-10-29','resolution_date':'2014-10-28','resolution_number':'30257',
  'evidence_summary':'Authorizes Peru president travel November5-16, including APEC bilateral meetings and a China working visit. Earlier planning context, not evidence of a November2 China-to-Xi intent expression or exact ANSA story identity.'},
 'R17_PERU_MINISTER_RESOLUTION_376':{
  'url':'https://www.mef.gob.pe/contenidos/servicios_web/conectamef_quechua/pdf/normas_legales_2012/NL20141104.pdf',
  'read_status':'actual_official_gazette_body_read_page7_lines817_864',
  'publication_date':'2014-11-04','resolution_date':'2014-11-03','resolution_number':'376-2014-PCM',
  'evidence_summary':'Peru foreign minister Gonzalo Gutierrez accompanies Peru president, including China APEC November10-12 and planned talks with Xi. Peru/Humala actor is not interchangeable with the benchmark China subject. November3 document does not establish November2 intent-expression date.'},
 'R17_PERU_TRADE_MINISTER_RESOLUTION_377':{
  'url':'https://www.mef.gob.pe/contenidos/servicios_web/conectamef_quechua/pdf/normas_legales_2012/NL20141104.pdf',
  'read_status':'actual_official_gazette_body_read_page8_lines954_1006',
  'publication_date':'2014-11-04','resolution_date':'2014-11-03','resolution_number':'377-2014-PCM',
  'evidence_summary':'Trade minister accompanies Peru president to the bilateral meeting with China president scheduled November12. Confirms planned meeting date, not exact November2 subject or expression.'},
 'R17_CHINA_MFA_HUMALA_MEETING':{
  'url':'https://www.mfa.gov.cn/esp/wjb/zzjg/ldmzs/xwlb/201411/t20141114_935001.html',
  'read_status':'actual_official_Spanish_article_body_read',
  'event_date':'2014-11-12',
  'evidence_summary':'Official account reports Xi-Humala talks on November12. A later actual meeting does not establish November2 intent expression, China rather than Peru subject, or identity with ANSA SpanishService Story35830096sentence3.'},
 'R17_PERU_APSN_REMOTE_ROUTES':{
  'receipt_file':'source-route-fetches-round17.json','read_status':'both_remote_fetches_failed; failures_preserved',
  'evidence_summary':'Public Peru blog feed network unreachable; known APSN July2008 digest SSL unexpected EOF. No fetched body or truth label. Web-readable July ThamrinAnanda context is another date, not S03 identity proof.'}
})
j=e['judgments']['S19']
j['stage']='detailed_source_review'
j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+['R17_PERU_TRAVEL_RESOLUTION_30257','R17_PERU_MINISTER_RESOLUTION_376','R17_PERU_TRADE_MINISTER_RESOLUTION_377','R17_CHINA_MFA_HUMALA_MEETING','R17_PERU_APSN_REMOTE_ROUTES']))
j['round17_source_note']='Official travel/meeting context now obtained. Nov2 ANSA story identity, China/Peru actor mapping and exact expression date remain unresolved; unchanged SouthKorea rawTop1 cannot decide whole-rank validity.'
j['rationale']+=' Round17: Primary Peru gazette and ChineseMFA body acquired; previously failed blog route retained as provenance. Context is insufficient to label exact dated gold or competitor.'
assert before=={k:tuple(j[x] for x in ('gold_truth','final_event_truth','reasonable_rank_adjustment')) for k,j in e['judgments'].items()}
assert len(e['judgments'])==40
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'sample_n':40,'new_detailed_review':'S19','truth_labels_changed':0,'training':False,'neural_rescoring':False}))
