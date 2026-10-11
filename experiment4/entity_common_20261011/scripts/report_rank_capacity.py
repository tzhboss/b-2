"""Disclose unchanged utility gates and costs relative to original QP controls."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root, cfg):
    path = root/cfg['original_context']
    assert sha(path) == cfg['original_context_sha256']
    context = json.loads(path.read_text())
    assert not context['test_read'] and not context['valid_read']
    assert context['status'] == 'COMPLETED_EXPLORATORY_ORIGINAL_CONTROL_TRAIN_COSTS'
    assert [row['seed'] for row in context['rows']] == cfg['seeds']
    checks = 0
    for name, digest in context['input_sha256'].items():
        assert sha(Path(name)) == digest, name
        checks += 1
    canonical = {}
    for experiment in [cfg['experiment'], cfg['utility_experiment']]:
        state = yaml.safe_load((root/'experiments'/experiment/'experiment.yaml').read_text())
        for name, digest in state['frozen_input_sha256'].items():
            assert sha(root/name) == digest, name
            checks += 1
        canonical[experiment] = state
    for row in context['rows']:
        for cost in row['original_top10_train_truth_costs'].values():
            assert cost['facts'] == 10*row['exempted_queries']
            assert cost['top10_loss'] == cost['top10_lost']/cost['facts']
            assert cost['excess_boundary_tie_queries'] == 0
    return context, canonical, checks


def report(root, cfg, context, canonical):
    source_can = canonical[cfg['experiment']]
    utility_can = canonical[cfg['utility_experiment']]
    assert not source_can['active_processes'] and 'terminal' in source_can['status']
    assert not utility_can['active_processes'] and 'terminal' in utility_can['status']
    assert utility_can['aggregation_exit_code'] == 0
    assert all(row['status'] == 'completed_audited' for row in source_can['jobs'].values())
    assert all(row['status'] == 'completed_audited' for row in utility_can['jobs'].values())
    source_cfg = json.loads((root/source_can['config']).read_text())
    utility_cfg = json.loads((root/utility_can['config']).read_text())
    path = root/utility_can['result_directory']
    summary = json.loads((path/'summary.json').read_text())
    source_path = root/source_can['result_directory']/'summary.json'
    source_summary = json.loads(source_path.read_text())
    assert not summary['unavailable'] and summary['original_gates_unchanged']
    originals = {row['seed']: row for row in context['rows']}
    endpoint = {row['id']: row for row in source_summary['rows']}
    contracts = {}
    for job in utility_cfg['jobs']:
        audit_path = root/utility_cfg['reaudit_directory']/job['id']/'audit.json'
        audit = json.loads(audit_path.read_text())
        assert not audit['failures'] and audit['status'] == 'PASS_FULL_COMMON_VALID_METRICS'
        costs = audit['capacity_contract_audit']
        assert costs['status'] == 'PASS_EXPLICIT_EXEMPTED_TRAIN_COST_REPLAY' and not costs['failures']
        for name, digest in costs['evidence_sha256'].items():
            assert sha(Path(name)) == digest
        for name, digest in audit['evidence_sha256'].items():
            assert sha(Path(name)) == digest
        contracts[(job['method'], job['seed'])] = costs
    records = []
    lines = ['# 固定容量豁免方案的完整收益与保护代价', '',
             '旧、新VALID各固定1024条开发查询；全部旧TEST74428／新TEST12406查询参加共同评价。三个种子完整后汇总，非最终留出、等成本SOTA或新颖性认证。', '',
             '| 方法 | 新VALID MRR | 旧VALID Top10掉榜% | 旧头／新答案MRR | 方向Hits10% | 固定收益结论 |',
             '|---|---:|---:|---:|---:|---|']
    for row in summary['rows']:
        method = row['method']
        assert row['seeds'] == cfg['seeds']
        values = row['values']
        avg = lambda name: statistics.mean(values[name])
        expected = dict(
            new_gate=avg('new_mrr') >= source_cfg['utility_new_mrr_ratio']*avg('source_new_mrr'),
            old_gate=avg('source_old_loss') > 0 and avg('old_loss') <= source_cfg['utility_old_loss_ratio']*avg('source_old_loss') and all(a <= b for a, b in zip(values['old_loss'], values['source_old_loss'])),
            new_answer_gate=avg('after_answer_raw_mrr') >= source_cfg['utility_direction_mrr_ratio']*avg('source_answer_raw_mrr') and avg('after_answer_hits10') >= source_cfg['utility_direction_hits10_ratio']*avg('source_answer_hits10'),
            exempt_train_gate=all(contracts[(method, seed)]['nonworse_than_source_exempt_train_gate'] for seed in cfg['seeds']),
            tie_gate=all(contracts[(method, seed)]['no_excess_boundary_ties_gate'] for seed in cfg['seeds']))
        for key, gate in expected.items():
            assert row[key] == gate
        decision = 'PASS_DIRECTIONAL_AND_EXPLICIT_TRAIN_COST_SCREEN' if all(expected.values()) else 'NO_GO_FOR_THIS_FIXED_COMBINATION'
        assert row['decision'] == decision
        lines.append(f"| {method} | {avg('new_mrr'):.6f} | {100*avg('old_loss'):.4f} | {avg('after_answer_raw_mrr'):.6f} | {100*avg('after_answer_hits10'):.4f} | {decision} |")
        paired_costs = []
        tests = []
        for seed in cfg['seeds']:
            audit = contracts[(method, seed)]
            before = originals[seed]['original_top10_train_truth_costs']
            actual = audit['controls']['after']['original_top10_train_truths']
            unprotected = audit['controls']['source']['original_top10_train_truths']
            assert audit['exempted_queries'] == originals[seed]['exempted_queries']
            assert unprotected == before['SourceSGD']
            qp = before[cfg['method_context_controls'][method]]
            assert actual['facts'] == qp['facts']
            gate = actual['top10_lost'] <= unprotected['top10_lost'] and actual['raw_mrr'] >= unprotected['raw_mrr']
            assert gate == audit['nonworse_than_source_exempt_train_gate']
            assert (actual['excess_boundary_tie_queries'] == 0) == audit['no_excess_boundary_ties_gate']
            paired_costs.append(dict(seed=seed, exempted_queries=audit['exempted_queries'],
                                     source=unprotected, original_qp=qp, after=actual,
                                     extra_train_top10_lost_vs_original_qp=actual['top10_lost']-qp['top10_lost'],
                                     train_mrr_delta_vs_original_qp=actual['raw_mrr']-qp['raw_mrr'],
                                     all_train_truths=audit['controls']['after']['all_train_truths_at_exempt_keys'],
                                     source_all_train_truths=audit['controls']['source']['all_train_truths_at_exempt_keys']))
            job = next(j for j in utility_cfg['jobs'] if j['method'] == method and j['seed'] == seed)
            test = endpoint[job['id']]
            assert test['old_after']['queries'] == 74428 and test['new_after']['queries'] == 12406 and test['candidates'] == 11258
            tests.append(dict(seed=seed, old_mrr=test['old_after']['raw_mrr'], new_mrr=test['new_after']['raw_mrr'],
                              old_top10_loss=test['old_top10_loss'], new_hits10=test['new_after']['hits10'],
                              source_inclusive_incremental_elapsed_s=test['training']['source_inclusive_incremental_elapsed_s']))
        records.append(dict(method=method, decision=decision, unchanged_registered_gates=expected,
                            fixed_valid_values=values, paired_original_qp_train_costs=paired_costs, complete_test=tests))
    lines += ['', '旧TRAIN事实代价使用固定认证查询上的原Top10真答案；下面的原QP比较是额外披露，不添加或替换038收益门槛。通过相对源SGD的门槛不意味着优于原QP保护。', '',
              '| 方法／seed | 原Top10 TRAIN事实数 | 源SGD掉榜数 | 原QP掉榜数 | 新方案掉榜数 | 相对原QP的MRR变化 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in records:
        for cost in row['paired_original_qp_train_costs']:
            lines.append(f"| {row['method']}／{cost['seed']} | {cost['after']['facts']} | {cost['source']['top10_lost']} | {cost['original_qp']['top10_lost']} | {cost['after']['top10_lost']} | {cost['train_mrr_delta_vs_original_qp']:+.8f} |")
    lines += ['', '豁免改变选定旧TRAIN事实与预测成员的保护合同；即使通过全部门槛，也不能称全部旧知识安全。JSON另列这些查询上的全部TRAIN事实及原来不在Top10的事实代价。原QP控制复用已有审计终态，未重新训练。', '',
              '既有失败、原032负结论与037/038冻结门槛保持不变。汇总读取完整三seed终态；不从部分seed挑选胜利。源SGD、范围准备和认证成本计入增量耗时；未计时的全局split完整性检查仍明确排除。']
    bound = {str(p.relative_to(root)): sha(p) for p in [root/cfg['original_context'], path/'summary.json', source_path, root/source_can['config'], root/utility_can['config']]}
    disclosure = dict(status='PASS_POST_TERMINAL_UNCHANGED_GATES_AND_ORIGINAL_QP_COST_DISCLOSURE',
                      rows=records, source_sha256=bound, registered_gates_changed=False,
                      final_heldout=False, novelty_claim=False, limits=['Original QP TRAIN cost comparison is context, not an added acceptance gate.', 'No source optimizer retraining or independent full TEST CPU rescore.'])
    assert not (path/'cost-disclosure.json').exists() and not (path/'report.md').exists()
    (path/'cost-disclosure.json').write_text(json.dumps(disclosure, indent=2))
    (path/'report.md').write_text('\n'.join(lines), encoding='utf8')
    print(disclosure['status'], path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', default='configs/rank-capacity-report.json')
    parser.add_argument('--stage', choices=['inputs', 'report'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = json.loads((root/args.config).read_text())
    context, canonical, count = inputs(root, cfg)
    if args.stage == 'report':
        report(root, cfg, context, canonical)
    else:
        output = root/cfg['input_review_output']
        assert not output.exists()
        output.write_text(json.dumps(dict(status='PASS_REPORT_INPUT_CONTEXT_AND_SOURCE_PINS_ONLY',
                                          checked_files=count, seeds=cfg['seeds'],
                                          original_top10_facts=[row['original_top10_train_truth_costs']['SourceSGD']['facts'] for row in context['rows']],
                                          original_context_sha256=sha(root/cfg['original_context']),
                                          candidate_outcomes_read=False, scientific_utility_asserted=False,
                                          reporter_sha256=sha(Path(__file__)), config_sha256=sha(root/args.config)), indent=2))
        print('REPORT_INPUT_CONTEXT_PASS', count, flush=True)
