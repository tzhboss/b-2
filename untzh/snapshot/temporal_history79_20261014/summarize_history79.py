"""Aggregate existing statistics and explicit history evidence, with unknowns."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
stats = json.loads((ROOT/'icews-history-summary.json').read_text())
candidates = json.loads((ROOT/'temporal-witness-candidates.json').read_text())
intervals = json.loads((ROOT/'interval-statistics.json').read_text())
evidence = json.loads((ROOT/'sources/history-evidence.json').read_text())
out = {'source':str(ROOT), 'primary_query_population_n':stats['n'],
       'models_retrained':False, 'sources':evidence, 'arms':{},
       'world_error_point_estimate':None,
       'world_reasonable_change_point_estimate':None,
       'reason_for_nonidentification':
       'No complete dated event truth, negative-event annotations, source documents, or retraction/correction ledger; an unrecorded event can be true or false.',
       'reasonable_future_expiration_for_fixed_date':
       'Inapplicable: later events cannot by themselves expire an earlier dated event. Correction/newly discovered contemporaneous truth is a separate unobserved process.'}

def incompatible(c):
    return (c['after_top1']=='Aam Aadmi Party' and c['fixed_date']<'2011-01-01'
            and (c['direction']=='head' or c['relation']=='Engage in negotiation'))

for arm, group_stats in stats['arms'].items():
    witnesses = [dict(c, evidence_id='AAP_FOUNDATION',
                      judgment='temporally incompatible final prediction under named-party identity')
                 for c in candidates[arm] if incompatible(c)]
    assert all(not c['after_true'] for c in witnesses)
    armout = {'benchmark_groups':group_stats,
              'temporal_incompatibility_witnesses':witnesses,
              'witness_search':'targeted organization lookup, not a random adjudication sample',
              'world_error_bounds':{}}
    for name,g in group_stats.items():
        def eligible(c):
            if name.startswith('anchor2005') and not c['fixed_date'].startswith('2005-'):return False
            if 'rank_worsening' in name and not c['rank_worsening']:return False
            if name=='all_rank_changes' and c['after_rank']==c['before_rank']:return False
            if 'original_top10_drops' in name and not(c['before_rank']<=10 and c['after_rank']>10):return False
            return True
        matched=[c for c in witnesses if eligible(c)]
        k=len(matched);n=g['n'];unrecorded=n-g['after_recorded_true_top1']
        assert k<=unrecorded
        armout['world_error_bounds'][name]={
            'n':n, 'confirmed_temporal_incompatibility_n':k,
            'incompatibility_with_before_recorded_true_n':sum(c['before_true'] for c in matched),
            'remaining_unrecorded_unknown_n':unrecorded-k,
            'final_recorded_true_n':g['after_recorded_true_top1'],
            'conditional_fraction_lower':k/n if n else None,
            'conditional_fraction_upper':unrecorded/n if n else None,
            'assumptions':'recorded positive answers are sound; selected exact party identity is correct; bounds concern final Top1 errors, not rank drift causes or externally verified before-answer truth',
            'no_positive_soundness_assumption_upper':1.0}
    out['arms'][arm]=armout

target=intervals['target_counts']
n=sum(target.values());complete=sum(v for k,v in target.items() if k.endswith('|both_known'))
out['wikidata_raw_coverage']={'leadership_records_n':n,'both_boundaries_known_n':complete,
                            'both_boundaries_known_fraction':complete/n,
                            'missing_boundary_n':n-complete,
                            'externally_confirmed_boundary_conflicts_n':2,
                            'conflict_sampling':'targeted Australia records, not a representative error-rate sample',
                            'scope':'raw TKBC Wikidata, not small TIE WIKI; no WIKI or YAGO model ranks measured'}
(ROOT/'history-interpretation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))

lines=['# 历史排名漂移与真实历史的核对结果','',
       '查询日期保持不变。2005–2014共81,772条方向查询；每条比较原年模型与2015最终模型。复用已保存的模型预测和排名，没有重训、独立审计、SHA门槛或归档。', '',
       '## 实际能回答的比例','',
       'TComplEx普通增量训练中，2005年8,514条查询有4,845条gold排名下降。以下所有比例的分母均为这4,845条；正确指第一名属于相同实体/关系/日期的标准split已记录答案集合。', '',
       '| 第一名的变化 | 数量 | 排名下降中的比例 |',
       '|---|---:|---:|']
anchor=stats['arms']['TComplEx_finetune']['anchor2005_rank_worsening']
labels={'recorded_true_to_recorded_true':'已记录正确 → 已记录正确',
        'recorded_true_to_unrecorded':'已记录正确 → 未记录（基准新增错误）',
        'unrecorded_to_recorded_true':'未记录 → 已记录正确',
        'unrecorded_to_unrecorded':'未记录 → 未记录（前后均未被基准支持）'}
for k,label in labels.items():
    v=anchor['transitions'][k]
    lines.append(f'| {label} | {v:,} | {100*v/anchor["n"]:.2f}% |')
lines += ['',
          f'195条“前后第一名均正确”中只有{anchor["changed_but_both_recorded_true_n"]}条更换了第一名实体（占排名下降的{100*anchor["changed_but_both_recorded_true_n"]/anchor["n"]:.2f}%）。这只能说明基准接受的答案切换，不能称为“后来现实变化使旧答案失效”。其余可能第一名没变而gold排名下降。', '',
          '排名下降与第一名对错是不同指标：第一名仍正确不代表整张排名没有损伤；未记录也不代表现实已证伪。没有把这四类当作现实合理/错误/未知的互斥真值标签。', '',
          '## 全部历史与对照','',
          '| 训练方式 | 历史排名下降数 | 其中已记录正确→未记录 | 占排名下降 | 前后第一名均正确 |',
          '|---|---:|---:|---:|---:|']
for arm,a in stats['arms'].items():
    g=a['all_rank_worsening'];c=g['transitions']
    lines.append(f'| {arm} | {g["n"]:,} | {c["recorded_true_to_unrecorded"]:,} | {100*c["recorded_true_to_unrecorded"]/g["n"]:.2f}% | {c["recorded_true_to_recorded_true"]:,} |')
lines += ['',
          '冻结历史控制保持旧候选分数和排名不变。它在2005年仍有3,570条排名下降，402条第一名从已记录正确变成未记录，确认新增候选竞争可以制造这种基准错误。这是机制证据，不能用它估计现实假事件比例。', '',
          '## 已核对的现实历史证据','',
          '[India Today / PTI的同期报道](https://www.indiatoday.in/india/north/story/arvind-kejriwal-aam-aadmi-party-formal-launch-jantar-mantar-122520-2012-11-25)明确记载Aam Aadmi Party在2012年11月26日正式成立。采用更保守的2011年以前截止，避免把筹建活动算作不存在；只标记作为事件主体的预测，或实际参加谈判的目标。声明中提及的对象、意向/请求目标及其他有前身/别名歧义的组织不纳入。', '',
          '| TComplEx普通训练的证据案例 | 固定日期 | gold排名 | 最终第一名 |',
          '|---|---|---:|---|']
ws=out['arms']['TComplEx_finetune']['temporal_incompatibility_witnesses']
for c in ws:
    query=(f'{c["query_entity"]} — {c["relation"]} → ?' if c['direction']=='tail'
           else f'? — {c["relation"]} → {c["query_entity"]}')
    lines.append(f'| {query} | {c["fixed_date"]} | {c["before_rank"]:g} → {c["after_rank"]:g} | {c["after_top1"]} |')
b=out['arms']['TComplEx_finetune']['world_error_bounds']['all_rank_worsening']
lines += ['',
          f'因此，在42,851条历史排名下降查询中，已找到{b["confirmed_temporal_incompatibility_n"]}条有外部成立史支持的最终时间不相容预测；其中{b["incompatibility_with_before_recorded_true_n"]}条原先第一名是已记录正确答案。它们是已发现案例，不是随机样本，也不能用“核对中的错误占比”推算全部。原先gold事件未逐条由原新闻独立确证，不能进一步声称已确认这些案例原先现实完全正确。', '',
          f'如果假定基准中已记录正例正确、上述实体身份无误，则排名下降群体的最终第一名现实错误比例只能限制在[{100*b["conditional_fraction_lower"]:.4f}%, {100*b["conditional_fraction_upper"]:.2f}%]；这是识别区间，不是置信区间。{b["remaining_unrecorded_unknown_n"]:,}条未记录预测仍待历史证据。若不接受已记录正例真实的假定，上界为100%。该区间不能叫UFR，因为它不证明所有旧答案的现实真实性，也不归因参数/候选。', '',
          '本次AAP定向核对在冻结历史控制中没有找到成立前预测；这不否定该控制已测得的其他新增候选干扰，也不代表其中所有未记录预测都是真实事件。各臂证据列表见history-interpretation.json。', '',
          '## 合理时间变化的定义需要修正','',
          '对固定2005年查询，未来2006–2015年发生的其他事件不能让2005年事件失效。因此，“旧答案因未来世界变化而过期”在这个任务中不适用；数据中也没有旧事件撤回/更正记录。不能把“没有过期标注”写成实测现实合理变化率0%。', '',
          '后续证据可能补充同一天的其他真答案或纠正原始错误，这可以支持合理的预测调整，但需要相同历史日期的证据和更正记录。当前只有静态事件表，无法知道这种情况占多少。', '',
          '对滚动当前时间，旧任期结束后新任期的答案变化才是有效时间变化。已读取澳大利亚议会[霍华德](https://handbook.aph.gov.au/parliamentarians/ZD4)和[陆克文](https://handbook.aph.gov.au/parliamentarians/83T)任职记录：交接日是2007-12-03。因此查询2007-06-01应为霍华德、2008-06-01应为陆克文；若始终查询2007-06-01，后来仍应为霍华德。这个历史核对不是WIKI模型结果。', '',
          '## WIKI有效区间不能直接当完整真值','',
          f'检查的是TKBC包的原始Wikidata文件，与TIE小型WIKI版本不同。P6/P35/P1308在2005–2015相交或缺边界的{n:,}条记录中，{complete:,}条（{100*complete/n:.2f}%）两侧边界已知；{n-complete:,}条（{100*(n-complete)/n:.2f}%）至少缺一侧。缺结束日期不能当永远有效，裁剪/填充的时间不是真实任期证据。', '',
          '实际发现两处原始日期冲突：霍华德的结束、陆克文第一次开始均写2007-03-12，而议会记录为2007-12-03。原始记录已保留、外部更正另存。这是针对性核对发现，不是2/全体记录的数据错误率；也不能把训练模型服从原始错误标签的行为算为真实历史遗忘。', '',
          '本轮没有WIKI/YAGO训练结果，也没有将原始Wikidata与小型WIKI混称。', '',
          '## 当前未完成项','',
          '全量现实合理变化/错误比例尚未识别。要得到可解释的点估计，需要具有代表性的历史查询样本、精确实体消歧、原始事件新闻或完整任期来源，并对合理/明确错误/未知逐条标注；未知需单列。当前证据已完成全量基准转换统计、具名历史反例和原始区间可用性检查，不能替代剩余历史标注。']
(ROOT/'历史变化核对结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'anchor2005':anchor,'world_bounds':b,'witnesses_by_arm':{
    arm:len(a['temporal_incompatibility_witnesses']) for arm,a in out['arms'].items()},
    'wikidata_raw_coverage':out['wikidata_raw_coverage']},ensure_ascii=False,indent=2))
