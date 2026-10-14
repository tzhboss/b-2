"""Recover S15 source context without equating publisher, quoted speaker and Obama."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=ROOT/'representative-history-evidence.json'
e=json.loads(p.read_text(encoding='utf8'))
before={k:(j['gold_truth'],j['final_event_truth'],j['reasonable_rank_adjustment']) for k,j in e['judgments'].items()}
e['sources'].update({
 'R16_PSAKI_AUG27_MIRRORED_TRANSCRIPT':{
  'url':'https://www.iranwatch.org/library/governments/united-states/executive-branch/department-state/daily-press-briefing-spokesperson-jen-psaki-excerpts-11',
  'read_status':'actual_attributed_StateDepartment_transcript_excerpt_body_read; original_official_fetch_unavailable',
  'event_date':'2014-08-27','named_speaker':'Jen Psaki',
  'evidence_summary':'The mirrored transcript gives Psaki answers to reporters about Iranian centrifuge reports and agreement compliance. It is not a statement by Obama; it does not exclude other Obama remarks that day. Neither speaker nor exact BBCMonitoringStory35266399 is matched to the original IranianMedia gold.'},
 'R16_IRAN_NUCLEAR_REUTERS_BODY':{
  'url':'https://www.brecorder.com/news/190911/iran-says-testing-new-nuclear-enrichment-machine-may-irk-west',
  'read_status':'actual_Reuters_body_via_BusinessRecorder_read; article displays ReutersCopyright2014',
  'publication_date':'2014-08-27','reported_speaker':'Ali Akbar Salehi','reported_outlet':'Fars News Agency',
  'evidence_summary':'Datedbody attributes centrifuge announcements to Salehi and reports Fars quoting him. Establishes contemporaneous Iranian nuclear reporting context, but does not identify which unnamed media person made the coded statement in BBCMonitoringStory35266399sentence3. Publishing an attributed quotation is not automatically the same CAMEO actor assertion.'},
 'R16_IRANWATCH_FARS_NEWS_BRIEF':{
  'url':'https://www.iranwatch.org/news-brief/aeoi-chief-iran-ready-launch-new-generation-nuclear-centrifuges',
  'read_status':'actual_IranWatch_editorial_news_brief_read; not originalFarsbody',
  'displayed_date':'2014-08-27','attributed_publication':'Fars News Agency',
  'evidence_summary':'Brief summarizes Salehi centrifuge announcement. Used only as attributed context, not original article or a proven named media-person statement.'},
 'R16_IRAN_ORIGINAL_OUTLET_FETCHES':{
  'urls':['https://www.presstv.ir/detail/2014/08/27/376767/iran-develops-ir8-centrifuges/',
          'https://www.tasnimnews.com/ar/news/2014/08/27/475526/'],
  'read_status':'PressTVweb_open_failed; Tasnimindexedexactarticle_web_open502; no_original_body',
  'evidence_summary':'Search previews and article URLs do not prove source text or actor mapping. No labels from failed fetches.'},
})
j=e['judgments']['S15']
j['stage']='detailed_source_review'
j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+['R16_PSAKI_AUG27_MIRRORED_TRANSCRIPT','R16_IRAN_NUCLEAR_REUTERS_BODY','R16_IRANWATCH_FARS_NEWS_BRIEF','R16_IRAN_ORIGINAL_OUTLET_FETCHES']))
j['round16_source_note']='Actual same-date mirrored Psaki transcript and Reuters report recovered. Psaki, quoted Salehi and Iranian media outlet are distinct actors; neither is automatically Obama or the coded unnamed IranianMedia person. OriginalBBCstoryidentity and exactCAMEOactor mapping unresolved. FinalObamaTop1unchanged, no whole-rank label.'
j['rationale']+=' Round16: Relatedactualbodyrecovered; speaker/source distinctions documented. Retaingold/finaltruthunknown rather than turning topiccontext into exactactor confirmation.'
assert before=={k:(j['gold_truth'],j['final_event_truth'],j['reasonable_rank_adjustment']) for k,j in e['judgments'].items()}
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'sample_n':40,'new_detailed_review':'S15','truth_labels_changed':0,'training':False,'neural_rescoring':False}))
