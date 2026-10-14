"""Extend dated competitor evidence using unchanged rank caches."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sources=json.loads(r'''{
  "TASMIN_FORMATION_PRIMARY": {
    "url": "https://www.tasnimnews.ir/fa/news/1398/08/22/2140097/کدخدایی-سالگرد-تأسیس-خبرگزاری-تسنیم-را-تبریک-گفت",
    "title": "Tasnim founding anniversary report",
    "read_status": "organization_Persian_article_body_read",
    "reported_founding_date_solar_hijri": "1391-08-22",
    "conservative_formation_year": 2012,
    "evidence_summary": "Actual article body explicitly states Tasnim was founded22Aban1391. Official English about page also describes2013 English launch about9months after Persian launch; distinct fromEnglishwebsite launch alone."
  },
  "TASMIN_ABOUT_PRIMARY": {
    "url": "https://www.tasnimnews.ir/en/about",
    "title": "Tasnim News Agency About us",
    "read_status": "organization_English_about_body_read",
    "evidence_summary": "English launch2013-08-09 about9months after Persian opening; corroborates2012 agency timeline but not day-precision founding from this page alone."
  },
  "AHRAR_FORMATION_MMP": {
    "url": "https://mappingmilitants.org/files/group-profiles/ahrar_al-sham.pdf",
    "title": "Mapping Militants Project Ahrar al-Sham",
    "read_status": "profile_PDF_summary_and_narrative_text_read",
    "conservative_formation_year": 2011,
    "evidence_summary": "Summary and narrative date founding2011, narrativelate2011 by amnestied Islamist prisoners. January1 in webstructuredform is not assumed exactday."
  },
  "JAYSH_PREDECESSOR_MMP": {
    "url": "https://mappingmilitants.org/mmp-group/liwa-al-islam",
    "title": "Mapping Militants Project Liwa al-Islam",
    "read_status": "profile_summary_article_text_read",
    "conservative_predecessor_formation_year": 2011,
    "merger_date": "2013-09-29",
    "evidence_summary": "Liwa al-Islam founded2011 afterAlloushreleased;2013 mergerformsJaysh al-Islam.2009predictions incompatible even if earlierpredecessor alias allowed."
  },
  "LEE_INTENT_KOREATIMES": {
    "url": "https://www.koreatimes.co.kr/southkorea/20101110/korea-russia-agree-to-boost-energy-ties",
    "title": "Korea, Russia agree to boost energy ties",
    "read_status": "contemporaneous_article_body_read",
    "publication_date": "2010-11-10",
    "publication_local_time": "19:05 KST",
    "evidence_summary": "Final body sentence explicitly reports Lee will meet leaders ofUK/Germany/US/China today. GermanyMerkel/USObama are candidate identities. Conditional support for announced intent as of publicationdate; today conflicts withactualThursdaymeeting context, and not a directlyquotedLeeexpression."
  },
  "LEE_OBAMA_AP_DATE": {
    "url": "https://6abc.com/archive/7779554/",
    "title": "Obama salutes US troops, condemns North Korea",
    "read_status": "AP_republication_article_body_read",
    "display_date": "2010-11-10",
    "evidence_summary": "Body includesThursdaymorningremarks, Wednesdaynightarrival and plannedmeetingLee; USdisplayNov10 differsSeoullocalNov11. Not taken as clean same-local-date evidence."
  },
  "LEE_OBAMA_CBS_DATE": {
    "url": "https://www.cbsnews.com/news/president-obamas-schedule-in-seoul-11-10-11-10/",
    "title": "President Obama's Schedule in Seoul",
    "read_status": "schedule_body_read",
    "display_date": "2010-11-10",
    "evidence_summary": "AlltimesEastern; Nov10EST eveningLee bilateral correspondsNov11Seoul. Supportsplannedpair, not sameSeouldate directLeeexpression. No longquotations, sourcewordlimit25."
  }
}''')
ep=ROOT/'representative-history-evidence.json'
e=json.loads(ep.read_text(encoding='utf-8'))
e['sources'].update(sources)
e['judgments']['S17']['evidence_ids']=list(dict.fromkeys(e['judgments']['S17']['evidence_ids']+['LEE_INTENT_KOREATIMES','LEE_OBAMA_AP_DATE','LEE_OBAMA_CBS_DATE']))
e['judgments']['S17']['rationale']+=' Later candidate review finds datedKoreaTimes reports Lee plannedUS/Germany meetings, with date/actor attribution qualifications; conditional filteringObama/Merkel gives8->6. Remaining5candidate truthsunknown. FinalJapan alreadyfiltered and stillunverified.'
ep.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=ROOT/'temporal-competitor-evidence.json'
z=json.loads(p.read_text(encoding='utf-8'))
cs=json.loads((ROOT/'fixed40-new-outrankers.json').read_text(encoding='utf-8'))
identity=json.loads((ROOT/'competitor-identity-context-round8.json').read_text(encoding='utf-8'))
rules={'Tasnim news agency':(2012,['TASMIN_FORMATION_PRIMARY','TASMIN_ABOUT_PRIMARY']),'Militant (Ahrar ash-Sham)':(2011,['AHRAR_FORMATION_MMP']),'Jaysh al-Islam':(2011,['JAYSH_PREDECESSOR_MMP'])}
existing={(w['sample_id'],w['entity_id']) for w in z['witnesses']}
new=[]
for rec in cs['records']:
 for c in rec['new_outrankers']:
  for name,(year,sids) in rules.items():
   if name not in c['fact'] or c['fact'][3]>=str(year)+'-01-01': continue
   assert identity['records'][name], name
   assert c['after_gap']>0
   assert c['before_gap'] is None or c['before_gap']<=0
   if (rec['sample_id'],c['entity_id']) in existing: continue
   w={'sample_id':rec['sample_id'],'candidate_name':name,**c,'gold_fact':rec['gold_fact'],'original_rank':rec['original_rank'],'final_rank':rec['final_rank'],'source_ids':sids,'gold_independent_support':e['judgments'].get(rec['sample_id'],{}).get('gold_truth')=='supported_at_query_semantic_granularity','label':'temporally_incompatible_under_named_organization_or_dated_affiliation_semantics','counterfactual_rank_reduction_if_this_candidate_removed':1,'qualification':'Organization identity from rawcountry/sector; members interpreted at eventdate, not retroactiveperson labels. Formationyear source derived, not TRAINfirstyear. Goldsoundness stillseparate.'}
   z['witnesses'].append(w); new.append(w)
z['temporal_witness_candidate_n']=len(z['witnesses'])
z['temporal_witness_query_ids']=sorted({w['sample_id'] for w in z['witnesses']})
z['temporal_witness_query_n']=len(z['temporal_witness_query_ids'])
z['temporal_witness_query_coverage']=z['temporal_witness_query_n']/40
z['known_error_rank_components_by_query']=dict(Counter(w['sample_id'] for w in z['witnesses']))
z['costs']['round8_raw_identity_seconds']=identity['seconds']
excluded={'sample_ids':['S33'],'candidate':'Social Democratic Party (China)','reason':'RawcountryChina, four2010records; not linked toTaiwanSDPfounded2015. Excludedthatformationmapping.'}
if excluded not in z['excluded_identity_ambiguous_witnesses']: z['excluded_identity_ambiguous_witnesses'].append(excluded)
z['round8_new_witness_n']=sum(w['candidate_name'] in rules for w in z['witnesses'])
p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s17=next(c for c in cs['records'] if c['sample_id']=='S17')
positive=[c for c in s17['new_outrankers'] if c['fact'][2] in ['Barack Obama','Angela Merkel']]
assert len(positive)==2 and all(c['after_gap']>0 and c['before_gap']<0 for c in positive)
partial={'scope':'Samefixed40; conditionalcandidate-supportscenario, not adjudicatedwhole-rankreasonablefraction or newmodelscore','sample_id':'S17','gold_fact':s17['gold_fact'],'original_rank':s17['original_rank'],'final_rank':s17['final_rank'],'candidate_evidence':[{'candidate':c,'source_ids':['LEE_INTENT_KOREATIMES'],'semantic_truth':'conditional_support_for_announced_intent_with_date_and_attribution_qualification'} for c in positive],'conditional_rank_after_supplemental_filter':s17['final_rank']-len(positive),'rank_drift_before':s17['final_rank']-s17['original_rank'],'conditional_rank_drift_after':s17['final_rank']-len(positive)-s17['original_rank'],'remaining_new_outrankers_unknown_n':5,'qualification':'Articlepublication19:05KSTNov10 and willmeetphrase;todaywordinconsistentwithactualNov11meeting and notdirectLeequotes. USdatedAP/CBSreflecttimezones. Onlyintentpredicate, noNov10Consultclaim. Nooriginalbenchmarkmodification.'}
(ROOT/'partial-positive-competitor-round8.json').write_text(json.dumps(partial,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'new_witness_n':len(new),'total_witness_n':len(z['witnesses']),'by_query':z['known_error_rank_components_by_query'],'S17_conditional_afterrank':partial['conditional_rank_after_supplemental_filter']},ensure_ascii=False))
