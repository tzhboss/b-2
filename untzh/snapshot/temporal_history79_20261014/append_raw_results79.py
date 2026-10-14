import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
raw=json.loads((ROOT/'raw-icews-summary.json').read_text())
base=json.loads((ROOT/'icews-history-summary.json').read_text())
interpretation=json.loads((ROOT/'history-interpretation.json').read_text())
interpretation['sources']=json.loads((ROOT/'sources/history-evidence.json').read_text())
assert raw['gold_exact_match_n']==base['n']==81772
comparisons={}
for arm,groups in raw['arms'].items():
    comparisons[arm]={}
    for name,g in groups.items():
        b=base['arms'][arm][name]
        # These are aggregate interpretation checks, not model rescoring.
        assert g['benchmark_recorded_before_not_in_current_raw_n']==0
        assert g['benchmark_recorded_after_not_in_current_raw_n']==0
        comparisons[arm][name]={
            'extra_before_recorded_n':g['before_full_recorded_true_n']-b['before_recorded_true_top1'],
            'extra_after_recorded_n':g['after_full_recorded_true_n']-b['after_recorded_true_top1'],
            'new_errors_recovered_n':g['benchmark_new_error_but_full_recorded_n']}
interpretation['full_raw_icews']={
    'version':raw['version'],'years':len(raw['years']),
    'records_n':sum(y['records_n'] for y in raw['years'].values()),
    'download_bytes':sum(y['bytes'] for y in raw['years'].values()),
    'gold_exact_match_n':raw['gold_exact_match_n'],
    'gold_exact_match_fraction':raw['gold_exact_match_fraction'],
    'comparisons':comparisons,
    'continuation_seconds':raw['seconds'],
    'cost_scope':'Continuation includes parsing all years and remaining downloads; excludes initial 2005/2006 downloads, original reader attempt and HTTP403 attempt, source reading and local mirror. Not total experiment time.',
    'source_columns':'Story ID, Sentence Number, Publisher available; article body/title/URL and negative-event labels absent.',
    'source_readme':'sources/ICEWS-Read-Me.pdf pages 1-2: Event Text is event-type label, not original sentence; records are automatically coded.',
    'does_not_prove':'unrecorded final predictions are not thereby proven false in reality',
    'retained_execution_history':[
        'raw_icews79_reader_v1.py was stopped during key preparation after identifying repeated NpzFile decompression; process was computing, not declared stuck. No model training or scoring occurred.',
        'raw-icews-optimized.log: 2005 parsed, 2006 urllib HTTP403; original failure retained.',
        '2006 downloaded with ordinary curl; continuation uses curl, materializes test arrays once and persists yearly matches. 2005 parsed once more because first attempt had no matched-tuple checkpoint; later years not duplicated.',
        'raw-icews-continuation.log: natural exit0, no hard timeout or independent audit.'
    ]}
(ROOT/'history-interpretation.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2))

lines=['## 完整原始ICEWS事件库的补充核对','',
       '已下载Harvard Dataverse [ICEWS Coded Event Data v37](https://doi.org/10.7910/DVN/28075)的2005–2014年度文件，按实体名、事件类型文本和日期精确匹配。没有模糊实体匹配、邻近日期匹配或重新计算神经模型排名。', '',
       f'共读取{interpretation["full_raw_icews"]["records_n"]:,}条原始记录，年度文件共{interpretation["full_raw_icews"]["download_bytes"]:,}字节。81,772条方向查询的gold全部精确匹配（100%）；正反两向可对应同一原始事实，因此不是81,772个独立事件。', '',
       '| 模型 | 全历史已记录正确→未记录 | 其中在完整原始库补回的预测 |',
       '|---|---:|---:|']
for arm,groups in raw['arms'].items():
    g=groups['all_historical']
    lines.append(f'| {arm} | {g["benchmark_new_error_n"]:,} | {g["benchmark_new_error_but_full_recorded_n"]:,} |')
lines += ['',
          '五臂在全部历史及排名下降群体中，第一名的前后“已记录正确”标记均与原TKG标准split一致，没有找到被基准遗漏、却在完整原始库记录的第一名预测。这是已考察预测集合上的完整匹配结果，不能外推原始库与TKG全部事实集合相同。', '',
          '具体到TComplEx普通训练：18,504条排名下降且已记录正确→未记录的查询，完整原始库补回0条；2005年对应2,652条，补回也是0条。不能把这些退化解释为本次精确匹配发现的基准子集遗漏。', '',
          '官方Read Me第1页明确Event Text是CAMEO事件类型名称，第2页说明Story ID是内部新闻编号、Sentence Number是句子位置。因此原始记录能定位媒体和句子编号，但不是原新闻正文，更不是现实负例真值。事件表中不存在某预测仍不构成现实证伪。', '',
          '已将7个AAP时间不相容案例的gold关联到原始Event ID/Story ID/媒体，保存witness-news-provenance.json；其中6个gold排名恶化、另1个gold排名改善。rank改善也可能仍预测现实不可能的主体，这再次说明rank方向与现实正确性不是一回事。', '',
          '找到2010年5月16日Times of India关于[Sajjan Kumar案件的同期报道](https://timesofindia.indiatimes.com/city/delhi/sajjan-to-face-trial-in-1984-riots-case/articleshow/5935822.cms)，支持法院参与，但报道说具体裁定在前一天，且内容涉及准许起诉；没有直接把这篇文章当作“May 16 / Investigate”的完整精确真值。这类事件日期与报道日期、编码关系与原句语义的差异也需保留。', '',
          f'接续程序{raw["seconds"]:.3f}秒；此数不含已下载的2005/2006文件、首版读取器和HTTP403失败阶段、来源阅读及镜像，不称总耗时。首版重复解压被修为一次载入，失败源码和日志保留；没有重新训练模型或重算原排名。', '',
          '全量现实合理/错误比例仍不能识别：缺少完整新闻真值和负例证据。已向用户询问是否有按Story ID可检索的新闻来源；期间完成了上述公共原始事件库核对。']
path=ROOT/'历史变化核对结果.md'
text=path.read_text()
marker='\n## 完整原始ICEWS事件库的补充核对\n'
if marker in text:text=text.split(marker)[0]
path.write_text(text+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(interpretation['full_raw_icews'],ensure_ascii=False,indent=2))
