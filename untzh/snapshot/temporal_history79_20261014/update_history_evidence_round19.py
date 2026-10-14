"""Publish paired fixed-truth results without promoting strict semantics to coder truth."""
import json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
e=read('representative-history-evidence.json');paired=read('paired-truth-drift-bounds-round19.json')
mainkey='organization_occurrence_external_positives_only'
personalkey='personal_action_occurrence_external_positives_only'
assert paired['summary'][mainkey]['certified_positive_ids']==['S38']
assert paired['summary'][personalkey]['certified_positive_ids']==['S33','S38']
labelkeys=('gold_truth','final_event_truth','reasonable_rank_adjustment')
oldlabels={k:tuple(j[x] for x in labelkeys) for k,j in e['judgments'].items()}
e['sources']['R19_WANG_ICELAND_OFFICIAL_MEETING']={
 'url':'https://gothenburg.china-consulate.gov.cn/xwdt/201406/t20140627_3644224.htm',
 'read_status':'actual_official_body_read_lines20_26','event_date':'2014-06-27',
 'evidence_summary':'Official account confirms WangYi met Icelandforeignminister and exchanged views on bilateral/international affairs. Does not give a statement or quoted remarks specifically aboutChina. Consult cannot automatically establish MakeStatement. Wang candidate truth remainsunknown.'}
e['sources']['R19_PAIRED_FIXED_TRUTH_CACHE']={
 'artifact':'paired-truth-drift-bounds-round19.json','read_status':'existing_scores_read; all40 filtered ranks reconstructed; endpoint-attaining fixed-truth assignments checked',
 'evidence_summary':'Same dated candidate has one truth variable across checkpoints. Persistent outrankers cancel. Previous endpoint-separate lower bound remains valid but can be unnecessarily loose. No neural scoring or new source labels.'}
j=e['judgments']['S33']
j['evidence_ids']=list(dict.fromkeys(j['evidence_ids']+['R19_WANG_ICELAND_OFFICIAL_MEETING','R19_PAIRED_FIXED_TRUTH_CACHE']))
j['round19_personal_occurrence_false_rank_drift_lower_bound']=1.0
j['round19_personal_occurrence_unjustified_rank_drift']='confirmed_only_if_query_requires_new_personal_action_on_that_date'
j['round19_scope_note']='No raw candidate formerly aboveQin falls belowQin. All three persistent raw outrankers cancel regardlessoftruth. Chiangcrossing contributes1 under newpersonalaction semantics, so totalfalse-rankdrift>=1 withoutassumingbenchmarkpositive soundness. Retrospectivecontent interpretation remainsunresolved; this scenario is separatefromorganization-only maincount.'
j['lifetime_candidate_note']='Conditionalonnewpersonalactionatquerydate, pairedfixedtruthbound nowprovespositivefalse-rankdrift>=1. Earlier separateendpoint bound was validbuttoo loose. ICEWScoderreportedcontent semantics remainsunresolved; no unconditionalcandidate/top1label.'
j=e['judgments']['S38']
j['round19_paired_false_rank_drift_lower_bound']=2.0
j['round19_paired_bound_file']='paired-truth-drift-bounds-round19.json'
j['round19_scope_note']='Fourknownfalsecandidatecrossingsminusatmosttwounknownfalseexits giveslower2; onepersistentbefore/afteroutrankercancels. Oldround10lower1retainedashistoricalvalidlooserbound.'
assert oldlabels=={k:tuple(j[x] for x in labelkeys) for k,j in e['judgments'].items()}
e['round19_paired_truth_bound_artifact']='paired-truth-drift-bounds-round19.json'
old=read('lifetime-crossing-interpretation-round12.json')
old['round19_paired_bound_clarification']={
 'artifact':'paired-truth-drift-bounds-round19.json',
 'S33_conditional_lower':1,'old_bound_status':'valid but loose; did not cancel persistent competitors',
 'interpretation':'S33 whole error drift now proven under personal-action occurrence semantics and independently supported gold; not an unconditional reported-content/coder label.'}
write('lifetime-crossing-interpretation-round12.json',old)
N=42851;n=40;start=time.perf_counter()
def tail(K,count):
 if K>N-n:return 1.0
 p0=math.prod((N-K-j)/(N-j) for j in range(n))
 if count==1:return 1-p0
 if count==2:return 1-p0-p0*n*K/(N-K-n+1)
 raise ValueError(count)
def lower(count):
 lo,hi=count,N-n
 while lo<hi:
  mid=(lo+hi)//2
  if tail(mid,count)>=.05:hi=mid
  else:lo=mid+1
 assert tail(lo-1,count)<.05<=tail(lo,count)
 return {'certified_n':count,'count_lower':lo,'fraction_lower':lo/N,'fraction_upper':1.0,
         'tail_previous':tail(lo-1,count),'tail_selected':tail(lo,count)}
confidence={
 'population_n':N,'sample_n':40,'confidence_level':.95,'point_error_fraction':None,'point_reasonable_fraction':None,
 'method':'One-sided finite-population hypergeometric tail inversion; at least certified positivecases in fixed random40. Conditional sound labels and common scenario semantics required. Posthoc exploratory, not preregisteredconfirmatory evidence.',
 'organization_occurrence':{**lower(1),'query_ids':['S38']},
 'personal_action_occurrence':{**lower(2),'query_ids':['S33','S38']},
 'conditions':'Fixeddatedtruth, acceptedindependentgold, datedorganizationidentity; personal scenario additionallyexcludeshistorical/posthumouspublication answers. No replacementor source-basedsampling. Scenario semanticvalidity cannotbe suppliedbyconfidencebounds.',
 'computation_seconds':time.perf_counter()-start,'training':False,'neural_rescoring':False}
write('paired-population-bounds-round19.json',confidence)
write('representative-history-evidence.json',e)
print(json.dumps({'main_ids':['S38'],'personal_occurrence_ids':['S33','S38'],'main_lower':confidence['organization_occurrence']['fraction_lower'],'personal_lower':confidence['personal_action_occurrence']['fraction_lower'],'source_truth_labels_changed':0},ensure_ascii=False))
