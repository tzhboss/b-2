"""Append round10 source reads and cached score bound; never rerun old updaters."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'representative-history-evidence.json'
e=json.loads(p.read_text(encoding='utf-8'))
e['sources'].update({
 'R10_CPLP_PUBLICATION':{'url':'https://files.dre.pt/1s/2006/06/117a00/43634365.pdf','read_status':'official_three_page_PDF_body_read_via_web_tool','evidence_summary':'PublishedJune20,2006; CPLP statutes amendments adoptedJuly30,2002 and approvedApril20,2006. Not same-day Angola-Portugal bilateral cooperation occurrence.'},
 'R10_ANGOLA_FINANCE':{'url':'https://www.historico.portugal.gov.pt/pt/o-governo/arquivo-historico/governos-constitucionais/gc17/comunicados-cm/cm-2007/20070222.aspx','read_status':'official_government_body_read','evidence_summary':'February22,2007 approval explicitly identifies bilateral credit convention signedLuandaApril5,2006. Cannot substituteJune20.'},
 'R10_ANGOLA_JUDICIAL':{'url':'https://diariodarepublica.pt/dr/detalhe/aviso/582-2006-655568','read_status':'official_notice_body_read','evidence_summary':'1995agreement, ratificationexchangeApril5,2006, effectiveMay5, noticepublishedMay11. Ongoingvalidcooperation is distinctfromdatedeventJune20.'},
 'R10_THAILAND_DPA':{'url':'https://www.brecorder.com/news/3700920/no-winners-as-thaksin-heads-out-from-cambodia-20091116987187','read_status':'contemporaneous_dpa_republication_body_read','publication_date':'2009-11-16','evidence_summary':'Bodycopyrightdpa, notAFP. RetrospectiveSaturdaydeparture andpriorweekdispute. China mentionedasregionalpowerseekingresolution, not targetofThai same-daycriticism. Noabsence-basednegative.'},
 'R10_NORWAY_TREATY':{'url':'https://www.regjeringen.no/no/dokumenter/stprp-nr-3-2007-2008-/id488644/?ch=1','background_url':'https://www.regjeringen.no/no/dokumenter/stprp-nr-3-2007-2008-/id488644/?ch=2','read_status':'official_chapters1and2_body_read','evidence_summary':'TreatysignedJuly11,2007; precedingnegotiationsMarch14-16/April24-26. DoesnotdirectlyestablishJune8personalConsult.'},
 'R10_STOLTENBERG_VG':{'url':'https://www.vg.no/nyheter/utenriks/i/JKBM8/inviterte-putin-til-norge','read_status':'contemporaneous_body_read','publication_date':'2007-06-09','updated_date':'2007-06-11','evidence_summary':'June9SaturdayPutinmeeting inStPetersburg; notJune8Fradkov orfull-day exclusionofotherFradkovmeetings.'},
 'R10_NLM_FETCH':{'url':'https://uzo.sakura.ne.jp/burma/nlm/nlm_data/nlm_2007/nlm_06_2007/nlm_13_06_2007.pdf','read_status':'local_PDF_body_extracted_and_page4_visually_read_after_web_size_rejection_and_remote_TLS_failure','publication_date':'2007-06-13','article_dateline':'Stockholm2007-06-11','inferred_event_date':'2007-06-08','page_1based':4,'source_attribution':'MNA/Xinhua cites Norwegian Broadcasting NRK','evidence_summary':'ContemporaneousarticleexplicitlystatesStoltenbergannouncedagreementinMoscowFridayaftermeetingFradkov. FridayprecedingMondayJune11isJune8, consistentwithindependentNTBtimeline. SupportsConsultgold atquerygranularity; notexactITARTassStoryIDlink, notdenialofotherFradkovmeetings.'},
 'R10_TNA_FETCH':{'url':'https://khmer-daily.blogspot.com/2009/','read_status':'fetch_failed_redirected_to_google_sorry','evidence_summary':'IndexedTNAtextnotusedfortruthlabels.'}
})
def append(sid,ids,text):
 j=e['judgments'][sid]
 j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+ids))
 j['rationale']=j['rationale'].split(' Round10:')[0]+' Round10: '+text
append('S11',['R10_CPLP_PUBLICATION','R10_ANGOLA_FINANCE','R10_ANGOLA_JUDICIAL'],'Knowncooperationdates2002/April5/May5 do notestablishJune20cooperationevent. Portugalsolecrossing staysunknown; noabsence-basednegative.')
append('S07',['R10_THAILAND_DPA','R10_TNA_FETCH'],'dpaNovember16bodyretrospective; Chinaregionalcontext nottargetofThaidatedcriticism. Bothgoldexactdate andChinacrossingunknown.')
append('S36',['R10_NORWAY_TREATY','R10_STOLTENBERG_VG','R10_NLM_FETCH'],'ActualJune13newspaperpage4bodyandlayoutread;Xinhua/NRKdescribesFridayMoscowagreementannouncementaftermeetingFradkov. DatelineMondayJune11makesFridayJune8, consistentwithNTBtimeline. GoldConsultsemanticallysupported, originalTop1samegoldsupported. NoexactITARTassStoryIDclaim. Bortnikov/Gryzlovtruthsstillunknown; no whole-rankerrorlabelfromgoldtruthalone.')
e['judgments']['S36']['gold_truth']='supported_at_query_semantic_granularity'
e['judgments']['S36']['original_top1_truth']='supported_at_query_semantic_granularity'
e['judgments']['S36']['stage']='detailed_source_review'
b=json.loads((ROOT/'false-rank-drift-bounds-round10.json').read_text(encoding='utf-8'))
for r in b['records']:
 r['gold_supported_at_query_semantic_granularity']=e['judgments'][r['sample_id']]['gold_truth']=='supported_at_query_semantic_granularity'
 r['guaranteed_error_drift_without_filter_soundness']=bool(r['gold_supported_at_query_semantic_granularity'] and r['world_false_rank_drift_lower_bound_without_benchmark_filter_soundness']>0)
 r['guaranteed_error_drift_conditional_on_filter_soundness']=bool(r['gold_supported_at_query_semantic_granularity'] and r['world_false_rank_drift_lower_bound_if_benchmark_filters_sound']>0)
assert b['raw_confirmed_query_ids']==[r['sample_id'] for r in b['records'] if r['guaranteed_error_drift_without_filter_soundness']]
b['source_label_refresh']='S36 independently supported after PDF body read; cached score bounds unchanged, no repeat score-array read or neural scoring.'
(ROOT/'false-rank-drift-bounds-round10.json').write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s38=next(r for r in b['records'] if r['sample_id']=='S38')
assert s38['guaranteed_error_drift_without_filter_soundness']
append('S38',[], 'Cachedrawbeforehas3aboveand0ties;after4knownfalseabove. Conditionalongoldsupportedanddatedorganizationidentity,false-competitorrankdriftatleast1, evenwithmaximallyfavorablemissingpositives. Strongerthancomponentexistence; no benchmarkfilter-soundness assumption. Samplelowerbound1/40,notpopulationestimate.')
e['judgments']['S38']['reality_corrected_false_rank_drift_lower_bound']=s38['world_false_rank_drift_lower_bound_without_benchmark_filter_soundness']
e['judgments']['S38']['unjustified_rank_drift']='confirmed_conditional_on_gold_semantics_and_dated_organization_identity'
e['judgments']['S38']['rank_bound_file']='false-rank-drift-bounds-round10.json'
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'source_n':len(e['sources']),'confirmed_error_drift_query_ids':b['raw_confirmed_query_ids'],'world_reasonable_fraction':None,'world_error_fraction':None},ensure_ascii=False))
