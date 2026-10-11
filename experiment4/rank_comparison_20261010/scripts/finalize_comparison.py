"""Promote checked components only; retain initial whole-audit failure."""
import argparse,csv,json,math
from pathlib import Path
import yaml
from run_rank_comparison import sha,verify

def main(root):
    manifest=verify(root);can=root/'experiments/KGE-RANK-COMP-001-AUDIT-V2/experiment.yaml'
    state=yaml.safe_load(can.read_text());assert state['status']=='terminal_pending_controller_review' and not state['active_processes']
    original=root/'experiments/KGE-RANK-COMP-001/experiment.yaml'
    assert yaml.safe_load(original.read_text())['status']=='failed_retained'
    cfg=json.loads((root/'config.json').read_text());table=[];audits={};failures=[]
    for model in cfg['models']:
        folder=root/'artifacts'/model;a=json.loads((folder/'audit-v2.json').read_text());term=json.loads((folder/'terminal.json').read_text())
        assert a['strictnull_discrepancies']==0 and a['old_vectors_byte_identical']==2800 and a['new_baseline_vectors_audited']==800
        for n,d in a['output_sha256'].items():assert sha(folder/n)==d
        audits[model]=a
        failures.append(dict(model=model,failed_checks=a['ridge_vector_gate_failures'],max_relative_vector_error=a['max_relative_vector_error'],max_ridge_normal_condition=a['max_ridge_normal_condition'],audit_sha256=sha(folder/'audit-v2.json')))
        rows=json.loads((folder/'per_case.json').read_text())
        for s in term['summaries']:
            if s['method']=='PublishedRidge':continue
            group=[x for x in rows if x['method']==s['method']];assert len(group)==400
            for key in ['success_at_k','no_damage','safe_success','damage_count','external_valid_damage','update_norm']:
                actual=math.fsum(float(x[key]) for x in group)/400
                assert abs(actual-s[key])<=1e-12*(1+abs(actual)),(model,s['method'],key)
            table.append(s)
    target=root/'results/KGE-RANK-COMP-001-AUDIT-V2';target.mkdir(parents=True,exist_ok=False)
    summary=target/'summary.csv'
    with summary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    receipt=dict(status='PASS_EIGHT_METHOD_COMPARISON_WITH_RIDGE_FAILURE_RETAINED',promoted_methods=[m for m in cfg['reused_methods']]+['StrictNull'],excluded_methods=['PublishedRidge'],
        input_manifest_sha256=sha(root/'input-sha256.json'),source_hashes={p.name:sha(p) for p in (root/'scripts').glob('*.py')},
        component_audits=audits,initial_failed_canonical_sha256=sha(original),summary_sha256=sha(summary),new_training=False,
        scientific_limitations=['Exposed400 VALID cohort; same public256d models and frozen-query candidate-column contract.',
          'No full entity shared-head/tail protection claim; no adaptation/training of continual or LM editors.',
          'All-split Support uses extra VALID/TEST optimization labels; both target conventions retained.',
          'Old QP KKT audit inherited; supplemental StrictNull independently checked; Ridge fails fixed vector gate.'])
    (target/'audit-summary.json').write_text(json.dumps(receipt,indent=2))
    (target/'ridge-failure-disposition.json').write_text(json.dumps(failures,indent=2))
    labels={'Direct':'Direct','PublishedRank':'Rank(.125)','SupportRegularized_TRAIN_adaptation':'Support TRAIN适配',
        'PredictionSetQP_RankNormCap':'TRUE+Prediction QP＋Rank','TrueOnlyQP_RankNormCap':'TRUE-only QP＋Rank',
        'SupportAllSplit_LiteralTargetEligible':'Support all-split：目标可选','SupportAllSplit_TargetExcluded':'Support all-split：排除目标','StrictNull':'StrictNull'}
    lines=['# 同一Rank协议的实测对照','',
      '两模型、各400个已暴露VALID请求、公开256维权重。无过滤全实体候选、strict >排名、Top10；同关系TRAIN原Top10真事实、顺序上限5000。SafeSucc＝目标进入Top10且此保护范围内无一旧正确答案掉出。这里只更新候选尾列，head查询冻结。', '',
      '复用034/036/037已审的5600个旧更新向量，在远程全部重新评价，旧目标排名和损伤字段逐例完全一致；新增800个StrictNull更新已核验。Ridge的800个更新保留，但独立向量门槛失败，不进入下面通过核验的性能表。没有训练、QP重新拟合、数据重选或参数搜索。','',
      '| 方法 | DistMult目标成功 | DistMult SafeSucc | ComplEx目标成功 | ComplEx SafeSucc |',
      '|---|---:|---:|---:|---:|']
    for m in cfg['reused_methods']+['StrictNull']:
        rr=[next(x for x in table if x['model']==k and x['method']==m) for k in cfg['models']]
        lines.append('| '+labels[m]+' | '+' | '.join(f'{100*x[key]:.2f}%' for x in rr for key in ['success_at_k','safe_success'])+' |')
    lines+=['','TRUE+Prediction QP＋Rank相对Rank的SafeSucc提高9.50/10.75个百分点；配对新增安全成功38/43例，原Rank安全成功丢失0例。目标成功率均为96%。这些是已知队列上的描述性比较，不是最终盲测、SOTA或全局安全结论。', '',
      'TRUE-only QP的SafeSucc为58.00%/59.75%，预测约束没有额外SafeSucc收益。TRUE+Prediction的外部旧事实损失/次为0.045/0.070，TRUE-only为0.1475/0.175，Rank为0.075/0.0925；这里外部VALID的部分查询已被TRAIN查询键覆盖，不能一概称未见查询泛化。', '',
      'Support TRAIN为适配；all-split是明确目标函数和目标处理约定的重建，含额外VALID/TEST标签。不能称精确复现作者最强完整方法或公平部署胜利。目标可选约定的0%结果保留，不据此称原论文有bug。', '',
      'StrictNull两模型旧事实无损率100%，目标成功率仅3%；它显示保护与可编辑性的取舍。Ridge初次审计在两个模型都失败；续审保持相对向量误差<1e-5、驻点残差<1e-7门槛不变，收集全部失败，不放宽门槛、不重新求解/选择生产结果。初次canonical仍为failed_retained，续审仅提升通过的八个方法。详细失败见results/KGE-RANK-COMP-001-AUDIT-V2/ridge-failure-disposition.json。', '',
      'IncDE/FastKGE/STARK/LKGE和KGEditor可以使用同样的raw排名与旧Top10保留指标，但原始架构、增量输入和训练信息与该单次编辑不同；它们此前的MRR/filtered排名结果未混入此表。直接同任务适配尚未完成，详见compatibility.md。', '',
      '之前041的真实共享head/tail、三目标、全关系合同仍是负结果：当前预测保护QP＋Rank的全局Joint SafeSucc两模型均2.5%。这与本表保护范围不同，不能用本表局部提升替代共享模型安全结论。', '',
      '执行位置：用户experiment4/rank_comparison_20261010。两CPU进程、各2个BLAS线程；本轮是向量重放与数值审计，未训练神经编辑器，不需要GPU。每模型生产重放耗时：'+', '.join(f"{k} {json.loads((root/'artifacts'/k/'terminal.json').read_text())['elapsed_s']:.2f}s" for k in cfg['models'])+'；该时间不是历史QP求解/训练成本，不能用于跨方法效率胜负。', '',
      '数值核查使用独立SVD/最小二乘、所有声明保护行的单列排名代数和抽样直接全候选重评分；不是独立科研同行评审。完整轻量审计与输入SHA绑定见results/目录。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state.update(status='completed_audited_qualified',audit=dict(status=receipt['status'],promoted_result_directory=str(target.relative_to(root)),initial_failure_retained=True,excluded_methods=['PublishedRidge']),active_processes=[])
    tmp=can.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(state,sort_keys=False));tmp.replace(can)
    (target/'canonical_after_review.yaml').write_bytes(can.read_bytes())
    verify(root)
    print(json.dumps(dict(status=state['status'],rows=len(table),ridge_failures=[dict(model=x['model'],count=len(x['failed_checks'])) for x in failures])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
