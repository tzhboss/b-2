"""One common table from already audited task rows; never invent missing scores."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
import numpy as np
import yaml


def build(root,config):
    cfg=json.loads((root/config).read_text());groups=defaultdict(list);bound={};checks=0
    for eid in cfg['audited_sources']:
        folder=root/'results'/eid
        canonical=yaml.safe_load((root/'experiments'/eid/'experiment.yaml').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        audit=json.loads((folder/'audit-summary.json').read_text())
        assert audit['jobs_total']==audit['jobs_passed']==len(summary['rows'])
        audits={a['id']:a for a in audit['audits']}
        for row in summary['rows']:
            assert canonical['jobs'][row['id']]['status']=='completed_audited'
            assert not audits[row['id']]['failures']
            assert not row['training']['test_read']
            assert row['candidates']==cfg['candidates']
            assert row['old_after']['queries']==cfg['old_queries']
            assert row['new_after']['queries']==cfg['new_queries']
            groups[(row['method'],row['kind'],row.get('dimension',200))].append(row)
            checks+=7
        for name in ['summary.json','audit-summary.json']:
            bound[str((folder/name).relative_to(root))]=sha256((folder/name).read_bytes()).hexdigest()
    records=[]
    for (method,kind,dim),rows in groups.items():
        rows=sorted(rows,key=lambda r:r['seed']);assert [r['seed'] for r in rows]==cfg['seeds'];checks+=1
        values=dict(old_mrr=[r['old_after']['raw_mrr'] for r in rows],new_mrr=[r['new_after']['raw_mrr'] for r in rows],
                    old_top10_loss=[r['old_top10_loss'] for r in rows],same_pool_loss=[r['same_pool_top10_loss'] for r in rows],
                    old_hits1=[r['old_after']['hits1'] for r in rows],new_hits1=[r['new_after']['hits1'] for r in rows],
                    new_hits10=[r['new_after']['hits10'] for r in rows],displacement=[r['mean_rank_displacement'] for r in rows],
                    incremental_training_seconds=[r['training']['elapsed_s'] for r in rows])
        records.append(dict(method=method,model=kind,dimension=dim,seeds=cfg['seeds'],job_ids=[r['id'] for r in rows],
                            metrics={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)),per_seed=v) for k,v in values.items()}))
    order=['LKGE','IncDE','FastKGE','STARK','SAGE','RandomExpansion','NewEntityOnly','FullFinetune','Replay']
    records.sort(key=lambda r:(order.index(r['method']) if r['method'] in order else len(order),r['model']))
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Partial audited development comparison; pending methods are not scored',candidate_count=cfg['candidates'],
                old_queries=cfg['old_queries'],new_queries=cfg['new_queries'],completed_jobs=sum(len(v) for v in groups.values()),
                assembly_checks=checks,source_sha256=bound,rows=records,pending_methods=cfg['pending_methods'])
    (out/'table.json').write_text(json.dumps(report,indent=2))
    lines=['# 新增实体共同任务：已审计阶段结果','',
           f"已完成 {report['completed_jobs']} 个方法/控制实验，种子17/29/43；未完成方法不填数值。当前是开发集比较，不能据此宣布SOTA或最终留出胜利。",'',
           '所有after均为11258候选、旧TEST74428和新TEST12406方向查询；无过滤，rank=1+count(score>gold)。模型维数差异如表公开。','',
           '| 方法 | 模型/维数 | 旧MRR | 新MRR | 旧Top10掉榜% | 同11258初态掉榜% | 新Hits10% | 平均位移 | 增量训练秒 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        m={k:v['mean'] for k,v in row['metrics'].items()}
        lines.append(f"| {row['method']} | {row['model']}/{row['dimension']} | {m['old_mrr']:.5f} | {m['new_mrr']:.5f} | {100*m['old_top10_loss']:.2f} | {100*m['same_pool_loss']:.2f} | {100*m['new_hits10']:.2f} | {m['displacement']:.2f} | {m['incremental_training_seconds']:.1f} |")
    for method in cfg['pending_methods']:lines.append(f'| {method} | 待完成 | — | — | — | — | — | — | — |')
    lines+=['','表中为三seed均值；各seed与样本标准差、Hits1在table.json。旧Top10掉榜以旧7505池成功事实为分母；同池辅助指标以扩展初态11258池成功事实为分母。平均位移是全旧查询的有符号位移，整体MRR改善不等于原Top10事实均被保留。','',
            'FastKGE/STARK的旧E/R不变仍出现掉榜，说明这些已实现适配中参数保留不能保证候选扩展后的排名保留。该结果不能外推为所有旧方法只保护分数。当前尚无本任务QP/Rank成绩，不宣称我们已胜出。','',
            '训练秒仅为增量工作进程耗时，未计旧底座、特征准备、审计与后续编辑器meta训练；架构和阶段成本不同，不据此宣布严格等成本效率胜负。IncDE的结构准备另有成本；SAGE的采样/映射在其增量耗时中。','',
            '实现边界：STARK作者reader实际消费距离列；IncDE作者loader实际覆盖排序文件，degree分支复用betweenness，均保留源码行为。SAGE旧producer的元数据完整性FAIL仍保留；本轮绑定scientific source、共同输入和推理重放通过，不能把它写成作者整个发布包完全认证。','',
            '审计是64完整候选行/任务的独立数值重放、查询/聚合/输入检查；不是独立重训优化器或对每个查询重建全评分矩阵。完整训练及评估数组保留在远程artifacts。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(completed_jobs=report['completed_jobs'],assembly_checks=checks,pending=cfg['pending_methods'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',default='configs/common-table.json');a=p.parse_args();build(a.root.resolve(),a.config)
