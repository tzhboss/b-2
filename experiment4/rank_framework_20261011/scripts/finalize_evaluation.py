"""Review all terminal audits, promote lightweight evidence, stop the round."""
import argparse,csv,datetime,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_scoring import sha

LABELS={'PublishedRank':'Rank (fraction .125)','PredictionSetQP_RankNormCap':'QP TRUE+Prediction + Rank','TrueOnlyQP_RankNormCap':'QP TRUE-only + Rank','SupportRegularized_TRAIN_adaptation':'Support TRAIN adaptation','SupportAllSplit_LiteralTargetEligible':'Support all-split literal','SupportAllSplit_TargetExcluded':'Support all-split exclude target'}
def finalize(root):
    c=root/'experiments/KGE-RANK-EVAL-001/experiment.yaml';state=yaml.safe_load(c.read_text());assert not state['active_processes'] and state['status']=='all_terminal_pending_controller_review'
    cfg=json.loads((root/'config.json').read_text());rows=[];expanded=[];reviews=[];failures=[]
    for job in cfg['jobs']:
        out=root/'artifacts'/job['id'];a=out/'audit.json';t=out/'terminal.json'
        receipt=state['jobs'][job['id']]
        if receipt['exit_code']!=0 or not a.exists() or not t.exists():
            failures.append(dict(id=job['id'],receipt=receipt));continue
        audit=json.loads(a.read_text());term=json.loads(t.read_text());assert audit['status'].startswith('PASS') and not audit['failures']
        assert audit['terminal_sha256']==sha(t)
        for name,d in term['output_sha256'].items():assert sha(out/name)==d
        for name,d in term['bound_input_sha256'].items():
            p=Path(name);p=p if p.is_absolute() else root.parent/'literature_baselines_20261010'/name
            assert sha(p)==d
        statuses='EXPLORATORY_UPSTREAM_FAIL_RETAINED' if job.get('exploratory') else audit['status']
        for row in term.get('summaries',[term.get('summary')]):
            assert row is not None;row=dict(row,id=job['id'],evidence_status=statuses,method_label=LABELS.get(row['method'],row['method']))
            rows.append(row)
        if 'expansion' in term:expanded.append(dict(term['expansion'],id=job['id'],evidence_status=statuses))
        reviews.append(dict(id=job['id'],audit=audit,terminal_sha256=sha(t),terminal_summary_sha256=sha(a)))
        receipt['status']='completed_audited_exploratory' if job.get('exploratory') else 'completed_audited_qualified'
    for n,d in state['frozen_input_sha256'].items():assert sha(root/n)==d,n
    results=root/'results/KGE-RANK-EVAL-001';results.mkdir(parents=True,exist_ok=False)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (results/'common-metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (results/'common-metrics.json').write_text(json.dumps(rows,indent=2))
    (results/'candidate-expansion.json').write_text(json.dumps(expanded,indent=2))
    (results/'audit-summary.json').write_text(json.dumps(dict(audits=reviews,failed_retained=failures,ridge='Excluded due to prior vector-audit failure; original preserved'),indent=2))
    def percent(x):return '—' if x is None else f'{100*x:.2f}%'
    lines=['# 统一 Rank 评价实测报告','', '已按无过滤、严格大于、Top10重算已有检查点和更新向量。所有方法使用同一指标定义；保留各自的模型、学习单元和保护查询集合。本轮无重新训练或QP求解，完成审计后停止。','',
      '同模型、同400编辑请求的Rank/QP对照可以直接作配对比较。持续学习/语言模型编辑的行用于检验其旧知识排名是否保持，不能据跨模型表格宣称统一任务SOTA。旧Top10掉出率以最初进入Top10的查询观察为分母，SafeSucc以完整编辑请求/批次为分母，两者不是同一个数。','',
      '| 方法 | 模型/数据 | 旧Top10数 | 掉出Top10 | 排名变化P95 | 新查询Hits10 | SafeSucc |','|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        marker='（探索，原完整性FAIL）' if 'EXPLORATORY' in row['evidence_status'] else ''
        lines.append(f"| {row['method_label']}{marker} | {row['model']}/{row['dataset']} | {row['old_top10']} | {percent(row['old_top10_loss_fraction'])} | {row['old_top10_delta_p95']} | {percent(row['new_hits10'])} | {percent(row['safe_success'])} |")
    lines+=['','排名变化P95统计所有旧Top10答案的有符号 after-before，可能为0即使一部分答案损坏。完整表另有仅退后样本的P95和最大位移。持续学习的新查询Hits10是最终快照TEST原始排名；KGEditor是暴露edit_test；Rank/QP是同400VALID请求，因此这些新查询率的难度不同。','',
      '## 候选扩张诊断','', '以下固定最终权重，将每个历史候选集合扩到最终实体全集。旧答案自身分数逐条不变；排名变化仅来自加入的竞争候选。这不等于同候选条件下的权重遗忘。','',
      '| 方法/数据 | 旧候选下Top10数 | 扩候选后掉出 | 排名变化P95 |','|---|---:|---:|---:|']
    for row in expanded:lines.append(f"| {row['method']}/{row['dataset']} | {row['old_top10']} | {percent(row['old_top10_loss_fraction'])} | {row['old_top10_delta_p95']} |")
    lines+=['','## 本轮能得出的结论','',
      '这些方法可以统一计算排名保护指标。结果不支持笼统断言“过去的方法都没有保护排名”：FastKGE/STARK在固定旧候选集合下确实保留全部旧排名；候选扩张带来的排名损伤是另一个可直接观察的条件。IncDE/LKGE等在固定候选下出现旧Top10掉出，说明其已有保护在这些观察上并不保证排名完全保持。']
    for kind in ['distmult','complex']:
        panel={x['method']:x for x in rows if x['id']=='rank-'+kind}
        if all(k in panel for k in ['PublishedRank','PredictionSetQP_RankNormCap','TrueOnlyQP_RankNormCap']):
            rk=panel['PublishedRank'];qp=panel['PredictionSetQP_RankNormCap'];true=panel['TrueOnlyQP_RankNormCap']
            lines.append(f"{kind}同400请求：新编辑Hits10均{percent(rk['new_hits10'])}，SafeSucc由Rank的{percent(rk['safe_success'])}到QP TRUE+Prediction的{percent(qp['safe_success'])}（增加{100*(qp['safe_success']-rk['safe_success']):.2f}个百分点）；逐旧知识观察掉出率由{100*rk['old_top10_loss_fraction']:.4f}%到{100*qp['old_top10_loss_fraction']:.4f}%。TRUE-only的SafeSucc为{percent(true['safe_success'])}，因此此轮未显示增加Prediction约束比TRUE-only更有效。")
    lines+=['','## 分数与排名','', '共同表包含正例分数变化除以编辑前候选分数标准差，及绝对标准化漂移≤0.001的旧Top10子集掉出率。阈值仅为描述性分箱。分数稳定同时排名损坏可证明这些观察上分数稳定不足以保护排名，不能据此把所有算法目标概括成“只保护分数”。','',
      '## 审计与边界','',
      '持续检查点：全部query重评分及向量/标量指标交叉核查；每个模型独立NumPy全候选复算960个状态/query，float64分数误差≤1e−12且排名精确一致。',
      'KGEditor：完整native批次文本/条件重建，所有before/after原native排序排名、Top3及logit（≤1e−4）核验；另算strict排名并CPU/GPU交叉检查。保留native共享embedding恢复限制，八个locality探针反复使用，不是完整独立L-Test；edit_test参与检查点选择。',
      'Rank/QP：原更新向量/检查点hash绑定，全部受保护事实排名重算并核对原400例损伤/成功率；抽样全候选标量复算。继承原QP证书，无新QP求解。仅冻结query单候选列、同关系TRAIN原Top10按顺序cap5000，不能称全局实体更新保护。all-split Support额外使用标签。Ridge原失败保留；SAGE仅探索。',
      '所有输入SHA在前后绑定，逐query产物留在远程artifacts且不进入Git。当前表是事后描述性评价，无新留出、新颖性或统计独立性声明。','', f'通过审核作业：{len(reviews)}/{len(cfg["jobs"])}；保留失败：{json.dumps(failures,ensure_ascii=False)}。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    state.update(status='completed_audited_qualified' if not failures else 'completed_partial_with_failures_retained',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),result_directory=str(results.relative_to(root)),failed_retained=failures,active_processes=[],next_round='not_started_user_requested_stop')
    c.write_text(yaml.safe_dump(state,sort_keys=False))
    manifest={str(p.relative_to(root)):sha(p) for p in [root/'report.md',c,*results.glob('*')] if p.is_file()};(root/'delivery-sha256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status=state['status'],rows=len(rows),expansion_rows=len(expanded),failed=failures),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();finalize(a.root)
