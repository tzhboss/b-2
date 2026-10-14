"""Summarize evidence progress without treating unresolved cases as errors."""
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sample=json.loads((ROOT/'representative-history-sample.json').read_text(encoding='utf-8'))
evidence=json.loads((ROOT/'representative-history-evidence.json').read_text(encoding='utf-8'))
lookups=json.loads((ROOT/'representative-search-receipts.json').read_text(encoding='utf-8'))
context=json.loads((ROOT/'sample-coded-event-context.json').read_text(encoding='utf-8'))
filter_check=json.loads((ROOT/'sample-filter-interpretation.json').read_text(encoding='utf-8'))
temporal_path=ROOT/'temporal-competitor-evidence.json'
temporal=json.loads(temporal_path.read_text(encoding='utf-8')) if temporal_path.exists() else None
rank_bound_path=ROOT/'false-rank-drift-bounds-round10.json'
rank_bounds=json.loads(rank_bound_path.read_text(encoding='utf-8')) if rank_bound_path.exists() else None
s38_component_n=temporal.get('known_error_rank_components_by_query',{}).get('S38',1) if temporal else 0
assert len(sample['records'])==sample['sample_n']==40
assert len({c['test_directional_index'] for c in sample['records']})==40
assert sum(len(b['queries']) for b in lookups['batches'][:13])==40
records=[]
for c in sample['records']:
    judgment=evidence['judgments'].get(c['sample_id'],{
        'stage':'initial_lookup_completed_no_verified_source_yet',
        'final_event_truth':'unresolved', 'original_top1_truth':'unresolved',
        'gold_truth':'unresolved', 'reasonable_rank_adjustment':'unresolved',
        'evidence_ids':[], 'rationale':'Initial search query completed; no exact event evidence adjudicated yet. Not a negative finding.'})
    assert all(s in evidence['sources'] for s in judgment['evidence_ids'])
    records.append({**c,**judgment})
supported=[c for c in records if c['final_event_truth']=='supported_at_query_semantic_granularity']
contradicted=[c for c in records if c['final_event_truth']=='explicitly_contradicted']
stages=Counter(c['stage'] for c in records)
review_n=stages['detailed_source_review']
blocked_n=stages['detailed_source_attempt_blocked']
context_n=stages['coded_context_identity_review']
initial_n=stages['initial_lookup_completed_no_verified_source_yet']
unknown_n=40-len(supported)-len(contradicted)
gold_supported=[c for c in records if c['gold_truth']=='supported_at_query_semantic_granularity']
gold_literal_date_counterindicated=[c for c in records if c.get('gold_literal_occurrence_truth')=='contradicted_under_actual_physical_visit_event_date_and_official_itinerary_coverage']
unchanged_n=sum(c['before_top1_fact']==c['after_top1_fact'] for c in records)
coverage_text=f'已为40条执行初步检索；{review_n}条读到相关或待排除正文并详细核对，{blocked_n}条详细来源获取失败，{context_n}条核对了原始编码中的身份歧义，{initial_n}条仍只有初步检索。正文核对不等于获得真假判定。'
report={
    'population_n':sample['population_n'],'sample_n':40,'sample_seed':sample['seed'],
    'sampling_design':sample['design'],
    'initial_lookup_n':40,'detailed_source_review_n':stages['detailed_source_review'],
    'detailed_source_attempt_blocked_n':stages['detailed_source_attempt_blocked'],
    'initial_only_unresolved_n':stages['initial_lookup_completed_no_verified_source_yet'],
    'coded_context_identity_review_n':context_n,
    'gold_supported_semantically_n':len(gold_supported),
    'gold_literal_physical_event_date_counterindication_n':len(gold_literal_date_counterindicated),
    'gold_literal_physical_event_date_counterindication_query_ids':[c['sample_id'] for c in gold_literal_date_counterindicated],
    'gold_literal_date_counterindication_note':'Conditional on actual occurrence semantics and official itinerary coverage; original report/coder semantics remains unresolved. Separate from final-answer truth and whole-rank error; not a population estimate.',
    'top1_unchanged_n':unchanged_n,
    'all40_original_coded_story_context_available':len(json.loads((ROOT/'all-sample-coded-context.json').read_text(encoding='utf-8'))['stories'])==40,
    'final_top1_supported_semantically_n':len(supported),
    'final_top1_supported_fraction_of_fixed_sample':len(supported)/40,
    'final_top1_explicitly_contradicted_n':len(contradicted),
    'final_top1_unresolved_n':40-len(supported)-len(contradicted),
    'unrecorded_but_semantically_supported_n':sum(not c['after_recorded'] for c in supported),
    'reasonable_rank_adjustment_confirmed_n':0,
    'world_reasonable_fraction':None,'world_error_fraction':None,
    'reality_corrected_rank_drift_bounds_progress':{k:v for k,v in rank_bounds.items() if k!='records'} if rank_bounds else None,
    'conditional_missing_positive_removes_benchmark_drift_n':filter_check['supported_missing_alternative_removes_drift_n'],
    'conditional_missing_positive_drift_fraction_of_fixed_sample':filter_check['fixed_sample_fraction_with_this_verified_conditional_filter_explanation'],
    'missing_positive_diagnostic_note':'One conditional filter explanation, not adjudicated reasonable whole-rank proportion; source mapping and original gold validity qualifications apply.',
    'caution':f'{len(supported)}/40 is current final-Top1 support coverage, not a reasonable-change estimate; zero confirmed contradictions does not mean zero real errors. {unknown_n} final-Top1 truths remain unresolved. Do not estimate rates only among source-accessible cases.',
    'finite_sample_partial_identification':{
        'final_top1_false_fraction_bounds':[len(contradicted)/40,(len(contradicted)+unknown_n)/40],
        'final_top1_supported_semantic_truth_fraction_bounds':[len(supported)/40,(len(supported)+unknown_n)/40],
        'whole_rank_reasonable_fraction_bounds':[0.0,1.0],
        'scope':'Finite fixed40 only, conditional on semantic mapping and source soundness; not population confidence intervals, whole-rank proportions, or newly introduced error rates.'},
    'definition':'Final Top1 truth at query semantic granularity and whole-rank justification are separate. Country-level semantic evidence does not assert exact automatic coder output.',
    'costs':{'sampling_seconds':sample['extraction_seconds'],'selected_story_scan_seconds':context['seconds'],
             'all_story_scan_initial_seconds':None,
             'cached_filter_interpretation_seconds':filter_check['seconds'],
             'all_story_scan_2014_continuation_seconds':json.loads((ROOT/'all-sample-coded-context.json').read_text(encoding='utf-8'))['current_invocation_seconds'],
             'diagnostic_scoring_and_identity_costs':temporal['costs'] if temporal else None,
             'scope':'Excludes browsing, code writing, source interpretation and mirror. No training. Initial provenance work had no scoring; subsequent Top5/all-crossing diagnostics rescored unchanged models.'},
    'temporal_competitor_progress':{k:v for k,v in temporal.items() if k!='witnesses'} if temporal else None,
    'records':records,
}
(ROOT/'representative-history-adjudication.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 随机历史查询核对进度','',
       '从TComplEx普通训练的42,851条历史gold排名下降查询中，无放回随机抽40条，seed20261014。全部保留；没有因为新闻难找换样。'+coverage_text, '',
       '| 当前证据状态 | 数量 / 固定40条 | 含义 |','|---|---:|---|',
       f'| 最终第一名有同日事件语义支持 | {len(supported)}/40 | 来源支持查询粒度事实，不代表整张排名合理 |',
       f'| 最终第一名已被明确证伪 | {len(contradicted)}/40 | 尚无样本中可作严格负例的结论，不表示错误率为0 |',
       f'| 现实真假尚未确定 | {unknown_n}/40 | 包含来源未取得、事件身份或精确日期未定 |', '',
       f'因此当前{len(supported)}/40只能称已取得支持证据的样本覆盖率。真正错误比例和整个排名变化合理比例仍缺完整判定。既有AAP六条时间错误见证是定向发现，不能混入这40条概率样本。', '',
       f'原gold有独立同日语义支持{len(gold_supported)}条；最终第一名未变{unchanged_n}条。原gold正确不自动证明新答案错误，第一名未变也不表示gold排名未受伤。', '',
       '本批40条中，最终第一名现实错误的逻辑识别范围仍为0%–95%，同日语义真答案范围为5%–100%；整张排序合理比例范围仍为0%–100%。这是未定项造成的有限样本界限，非总体比例、非置信区间、非新增错误率。只在能找到新闻的条目上计算比例会产生来源可得性偏差。', '',
       '## 本轮新增历史证据','',
       'S02：2013-09-17阿巴斯与美国特使Martin Indyk会谈，[同日IANS转新华社正文](https://www.business-standard.com/amp/article/news-ians/palestine-seeks-more-us-role-in-peace-talks-113091700936_1.html)支持原gold；与Obama会面写为下周安排，没有排除当天另有与Netanyahu的磋商。', '',
       'S11：[外交部会见报道](https://www.mfa.gov.cn/gjhdq_676201/gj_676203/fz_677316/1206_677390/xgxw_677396/200606/t20060621_8015169.shtml)明确2006-06-20已签合作协议，[引渡条约报道](https://www.mfa.gov.cn/web/wjb_673085/zzjg_673183/tyfls_674667/xwlb_674669/200606/t20060622_7669516.shtml)明确同日实际签署，支持安哥拉与中国合作。最终葡萄牙未知。', '',
       'S37：[AFP同日通讯正文](https://www.lorientlejour.com/article/901801/cinq-pays-du-sahel-appellent-lonu-a-intervenir-en-libye-en-accord-avec-lua-sommet-a-nouakchott.html)记载2014-12-19毛里塔尼亚等五国向安理会呼吁在利比亚部署国际力量，支持原gold。最终Mogherini是否另有呼吁未知。', '',
       '40条原gold故事的全部编码上下文已取到（不是新闻全文）。S03 Ananda被编码为缅甸佛教人物，不能套用印尼赛车手同名新闻；S09同一句编码多个国家。另读到的Lenku旅游报道日期为2014-06-02，不能支持3月31日；Papandreou党内竞选文章也未取得所需会谈证据。', '',
       '## 答案真实性与filtered排名下降的具体联系','',
       '读取远程既有评分缓存和evaluate76.py：原始Top1尚未过滤其他正确答案，但gold排名已经按同日全部split正例过滤。因此S33的习近平已被过滤，他当天确实讲话也不能解释秦刚filtered rank3→62。不能把“真实最终Top1”直接当“合理排名下降”。', '',
       'S39的伊拉克未被基准过滤；缓存分数伊拉克14.516507、gold中国13.873884，伊拉克唯一raw Top1，gold平均并列排名为2。一个严格超过gold的候选已占去全部排名增量。在接受报道的国家级Consult语义对应、并仍保留原gold作为评价目标的条件下，补充过滤伊拉克会将after rank2变为1，与before rank1相同。原模型、基准和排名文件未改写，没有新评分。', '',
       '因此40条中现有1条（2.5%）具备“补入有同日语义证据的漏标答案后，基准排名下降消失”的条件解释。它不是总体合理率：原gold在精确日期的真实性未独立确定，其他38条最终答案仍未知，整个排序与模型更新合理性也没有被证明。详见sample-filter-interpretation.json。', '',
       '## 新证据：基准未记录的答案可能现实成立','',
       'S39固定2009-04-19，查询“谁与也门磋商？”：原第一名中国、最终伊拉克，gold中国rank1→2。最终伊拉克不在标准split或完整原始事件表的精确答案中。', '',
       '[SABA经OANA发布的原报道](https://oananews.org/node/56303)正文明确April19/Sunday在萨那举行了伊拉克和也门部长的技术教育会谈；网站发布于April20。它在国家代表的查询粒度上支持伊拉克—磋商—也门，不能因为精确事件表缺失就判现实为假。新增读取2009原始年度文件也没有同日Iraq/Yemen country-pair事件；没有更改原始匹配结果或补写基准。', '',
       '原gold的Story10806432含中国检察长访问；[外交部原报道](https://www.mfa.gov.cn/gjhdq_676201/gj_676203/yz_676205/1206_677124/1206x2_677144/200904/t20090419_9306365.shtml)给出4月18日会谈、17–19日访问。尚未证明该网站报道对应内部Story ID，也未排除19日另有中也会谈。因此不直接宣布原gold错误或整个排名调整合理。', '',
       'S33固定2014-06-27，查询“谁就中国发表声明？”：原第一名王毅、最终习近平，gold秦刚rank3→62。新华社[边海防会议报道](https://www.ecns.cn/2014/06-28/121350.shtml)正文与照片日期支持习近平当天讲话；[官方秦刚记者会实录](https://losangeles.china-consulate.gov.cn/eng/confenrence/201406/t20140628_4992619.htm)也支持gold。最终第一名真实，不等于gold严重下降的整个排名都合理。', '',
       '## 原始日期和关系也需要来源核对','',
       'S10的最终第一名与原第一名均为“马来西亚在2010-12-13接待李明博”，不是新产生的第一名错误。韩国[官方12月14日说明](https://www.mofa.go.kr/eng/brd/m_5679/view.do?seq=309607)将该次印尼/马来西亚行程限定为前周四至周六（9–11日），[同期桥梁开幕报道](https://www.koreatimes.co.kr/southkorea/20101213/fixed-link-connects-busan-geoje-island)记载13日在韩国的活动。该已识别访问日期有反证，但没有完整行程或精确原新闻身份，保留“有日期反证的未知”。', '',
       'S04的gold为Lucy Kibaki在2005-05-16对Nation Media Group暴力抗议，最终预测Raheel Sharif。[同期新闻](https://mg.co.za/news/south-africa/2005-05-04-president-s-wife-storms-newspaper/)和[判决文本镜像](https://sheriahub.com/cases/ke/caselaw/otieno-clifford-richard-v-republic-2006-eklr-9SYFm)把已识别事件日期写为5月2–3日，16日则是私人起诉申请。原raw同Story sentence2为起诉、sentence5为riot，提示回述事件被同一日期编码；尚未确证原AFP全文/内部ID。官方判决页面403，实际读的是镜像。不能仅凭最终预测很荒谬，就把严格负例标注补成已证伪。', '',
       'S35最终和原第一名都为Obama—Consult—Netanyahu，固定2010-05-26。[当日JTA报道](https://www.jta.org/2010/05/26/united-states/emanuel-invites-netanyahu-to-white-house-meeting)实际是Emanuel送达邀请、Obama会面计划在6月1日；不能将报道日期当实际会面日期，也未排除当天另有电话磋商。保持未知。', '',
       '## 全部抽样条目','',
       '| ID | 最终预测事实（主体 / 关系 / 客体 / 日期） | gold排名 | 外部证据状态 |','|---|---|---:|---|']
for c in records:
    fact=' / '.join(c['after_top1_fact']).replace('|','\\|')
    label='同日语义支持' if c in supported else ('详细核对仍未知' if c['stage']=='detailed_source_review' else ('正文获取失败，未知' if c['stage']=='detailed_source_attempt_blocked' else ('编码身份待消歧' if c['stage']=='coded_context_identity_review' else '已初检，待深入核对')))
    lines.append(f"| {c['sample_id']} | {fact} | {c['original_gold_rank']:g}→{c['final_gold_rank']:g} | {label} |")
lines += ['',
          f"抽样读取{sample['extraction_seconds']:.3f}秒；三年度选定原故事上下文读取{context['seconds']:.3f}秒。费用不含网页检索/解释/镜像，非总耗时。无新训练或独立审计；后续Top5与全新增竞争者诊断对未修改checkpoint重新评分，成本另列。", '',
          '补全40条上下文的首版仅支持zip，在2014未压缩tab处失败后保存进度；修复后只接续2014。2.816秒仅该接续耗时，2005–2013初段完整耗时未单独保存（未知，非0）。', '',
          '目标仍未完成。下一需要继续这批固定样本的正文、实体消歧和精确日期核对；新闻中不提及某预测不能单独当负证据。']
if temporal:
    lines += ['', '## 后续新增：实际竞争者的历史证据', '',
              f"全部{temporal['new_strict_outranker_total_n']}个新超过gold的候选已提取；{temporal['temporal_witness_candidate_n']}个有组织时间反证，涉及{temporal['temporal_witness_query_n']}/40条。按指定组织/当时成员语义解释时成立，不能写成总体错误率。", '',
              f'S38同日SITA/ČTK报道支持斯洛伐克外交部批评匈牙利外交部；后来才成立的组织或成员类别在2009年查询中新增并超过gold。当前已确认{s38_component_n}个错误排名位置，移除后after rank162→{162-s38_component_n}；gold原rank4，其余位置仍未判定。1/40是固定样本错误成分的条件下限，不是完整UFR。S21/S04原gold同日真实性未定。', '',
              'CNS同名泰国机构raw记录早于2006政变，未完成身份消歧，排除严格见证。各来源/分差/身份条件及成本见[错误排名成分核对](错误排名成分核对.md)及temporal-competitor-evidence.json。']
if (ROOT/'round4-additional-source-notes.md').exists():
    lines += ['', '## 补充会谈来源', '',
              (ROOT/'round4-additional-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()]
if (ROOT/'round5-source-notes.md').exists():
    lines += ['', '## 第五轮补充来源', '',
              (ROOT/'round5-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()]
if (ROOT/'round67-source-notes.md').exists():
    lines += ['', '## 第六、七轮补充来源', '',
              (ROOT/'round67-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()]
if (ROOT/'round8-source-notes.md').exists():
    lines += ['', '## 第八轮逐候选证据', '',
              (ROOT/'round8-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()]
if (ROOT/'round9-source-notes.md').exists():
    lines += ['', '## 第九轮实际发生日期与目标语义', '',
              (ROOT/'round9-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()]
if (ROOT/'round10-source-notes.md').exists():
    lines += ['', '## 第十轮错误排名漂移下限', '',
              (ROOT/'round10-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()]
(ROOT/'随机历史查询核对进度.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
interpretation_path=ROOT/'history-interpretation.json'
interpretation=json.loads(interpretation_path.read_text(encoding='utf-8'))
interpretation['representative_sample_progress']={k:v for k,v in report.items() if k!='records'}
interpretation['representative_sample_evidence_file']='representative-history-evidence.json'
interpretation['historical_bounds_note']='Earlier witness bounds are retained as initial evidence bounds, not a final adjudicated estimate. New country-level article support is recorded separately; positive soundness remains an explicit assumption, with source-date counterevidence now identified in the sample.'
interpretation_path.write_text(json.dumps(interpretation,ensure_ascii=False,indent=2),encoding='utf-8')
main_path=ROOT/'历史变化核对结果.md'
main_text=main_path.read_text(encoding='utf-8')
marker='\n## 固定随机样本的新增外部核对\n'
main_text=main_text.split(marker)[0]
main_text+=marker+'\n'+coverage_text+f'最终Top1有同日查询语义支持{len(supported)}条，明确证伪{len(contradicted)}条，尚未确定{unknown_n}条。支持覆盖率不能写合理变化率，零证伪不能写真实错误率0%。整张排名合理/错误比例仍未识别。\n\n'
main_text+=f"独立同日语义证据支持原gold{len(gold_supported)}条（{'/'.join(c['sample_id'] for c in gold_supported)}）；全40原故事编码上下文已取得。原gold真实不自动否定最终候选。最终Top1未变{unchanged_n}条，需要分别评估答案真假与排序退化。\n\n"
main_text+='评分缓存解释新增：S33习近平已经被filtered评价排除，其真实讲话不能解释秦刚rank3→62；S39伊拉克是未过滤且唯一超过gold的候选，在接受同日语义映射的条件下，补充过滤它使中国rank1→2恢复1→1。40条中1条有此条件漏标解释，不是总体合理率或完整现实误差判定。\n\n'
main_text+='S39（2009-04-19）最终伊拉克与也门磋商未被基准或完整原始事件表精确记录，但SABA正文支持当天两国部长会谈。S33（2014-06-27）习近平声明有同日新闻支持、gold秦刚也有官方记者会支持。未记录不等于现实错误已获得实际同日事件反例；不能仅据最终答案真实宣布gold排名下降合理。\n\n'
main_text+='已识别访问/抗议的报道日期与事件日期有反证（S10/S04），未直接将疑似事件身份或局部行程当完整负例。详见[随机历史查询核对进度](随机历史查询核对进度.md)、representative-history-adjudication.json和representative-history-evidence.json；现有条件识别区间保留为早期证据界限，非最后比例。\n'
if temporal:
    main_text+=f'\n累计取得{temporal["temporal_witness_candidate_n"]}个有组织时间反证的新竞争者，涉及S04/S21/S38三条；S38原gold有同日正文支持，已确认{s38_component_n}个错误排名位置。因此固定40样本有1条包含已证实的错误成分（依指定组织/当时成员身份语义），2.5%是样本条件下限，不是总体UFR。其他位置/条目未知；CNS及中国社会民主党身份歧义排除严格见证。此前两次诊断评分耗时0.943/1.031秒，后续只读既存评分结果；两轮身份原始上下文读取8.811/6.775秒。详见[错误排名成分核对](错误排名成分核对.md)。\n'
main_path.write_text(main_text,encoding='utf-8')
if (ROOT/'round67-source-notes.md').exists():
    main_path.write_text(main_text+'\n## 后续日期与关系核对\n\n'+(ROOT/'round67-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()+'\n',encoding='utf-8')
if (ROOT/'round8-source-notes.md').exists():
    main_path.write_text(main_path.read_text(encoding='utf-8')+'\n## 第八轮实际竞争者证据\n\n'+(ROOT/'round8-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()+'\n',encoding='utf-8')
if (ROOT/'round9-source-notes.md').exists():
    main_path.write_text(main_path.read_text(encoding='utf-8')+'\n## 第九轮实际发生日期与目标语义\n\n'+(ROOT/'round9-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()+'\n',encoding='utf-8')
if (ROOT/'round10-source-notes.md').exists():
    main_path.write_text(main_path.read_text(encoding='utf-8')+'\n## 第十轮错误排名漂移下限\n\n'+(ROOT/'round10-source-notes.md').read_text(encoding='utf-8').split('\n',1)[1].strip()+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
