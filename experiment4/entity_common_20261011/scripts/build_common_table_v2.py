"""Assemble audited rows plus explicit per-seed pending/failure status."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from hashlib import sha256
import numpy as np
import yaml


def build(root,config):
    cfg=json.loads((root/config).read_text());groups=defaultdict(list);states=defaultdict(list);bound={};checks=0;corrections={}
    source_ids=cfg['sources']+cfg.get('continuation_sources',[]);selected={};attempts=defaultdict(list)
    def slot(job):return (job['method'],job['kind'],job['seed'],job.get('after_dimension',job.get('dimension',768 if job['kind']=='EXBert' else 200)))
    for eid in source_ids:
        path=root/'experiments'/eid/'experiment.yaml';can=yaml.safe_load(path.read_text());source_cfg=json.loads((root/can['config']).read_text())
        for job in source_cfg['jobs']:
            key=slot(job)
            if key in selected:assert eid in cfg.get('continuation_sources',[]),'Undeclared duplicate experiment slot'
            selected[key]=(eid,job['id']);attempts[key].append(dict(experiment=eid,id=job['id'],**can.get('jobs',{}).get(job['id'],dict(status='queued'))))
    utility=None
    scope_diagnostic=None
    shared_scope=None
    if cfg.get('shared_guard_scope_source'):
        path=root/'experiments'/cfg['shared_guard_scope_source']/'experiment.yaml';can=yaml.safe_load(path.read_text())
        assert can['status']=='completed_audited_artifact_checks_only' and not can.get('active_processes')
        result=root/can['result_directory']/'summary.json';raw=json.loads(result.read_text());assert raw['jobs_passed']==raw['jobs_total']==len(cfg['seeds'])
        columns=['seed','source_true_rows','true_initially_retained','true_initially_ineligible','source_prediction_queries',
                 'prediction_initially_retained_queries','prediction_initially_ineligible_queries','affected_old_train_query_keys',
                 'full_reference_members','full_grouped_queries','scoped_reference_members','scoped_grouped_queries','full_candidate_scoring_s','elapsed_s','peak_gpu_memory_bytes']
        shared_scope=[]
        for item in raw['rows']:
            assert not item['audit']['failures'] and item['audit']['complete_reference_scope_reconstructed'] and item['audit']['affected_keys_train_only']
            entry={name:item['result'][name] for name in columns}
            assert entry['true_initially_retained']+entry['true_initially_ineligible']==entry['source_true_rows']
            assert entry['prediction_initially_retained_queries']+entry['prediction_initially_ineligible_queries']==entry['source_prediction_queries']
            shared_scope.append(entry)
        assert sorted(r['seed'] for r in shared_scope)==cfg['seeds']
        for src in (path,result):bound[str(src.relative_to(root))]=sha256(src.read_bytes()).hexdigest()
    if cfg.get('protection_scope_source'):
        path=root/'experiments'/cfg['protection_scope_source']/'experiment.yaml';can=yaml.safe_load(path.read_text())
        assert can['status']=='completed_checked_train_scope_diagnostic_only' and not can['active_processes']
        result=root/can['result_directory']/'summary.json';scope_diagnostic=json.loads(result.read_text())
        assert scope_diagnostic['status']=='PASS_TRAIN_SCOPE_DIAGNOSTIC_ONLY' and not scope_diagnostic['test_read']
        for src in (path,result):bound[str(src.relative_to(root))]=sha256(src.read_bytes()).hexdigest()
    additional_utility=[]
    for eid in cfg.get('additional_utility_sources',[]):
        path=root/'experiments'/eid/'experiment.yaml';can=yaml.safe_load(path.read_text());bound[str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
        item=dict(experiment=eid,status=can['status'],aggregation_exit_code=can.get('aggregation_exit_code'),result=None)
        if can.get('aggregation_exit_code')==0 and not can.get('active_processes'):
            result=root/can['result_directory']/'summary.json';item['result']=json.loads(result.read_text());bound[str(result.relative_to(root))]=sha256(result.read_bytes()).hexdigest()
        additional_utility.append(item)
    precision=None
    if cfg.get('foundation_precision_source'):
        path=root/'experiments'/cfg['foundation_precision_source']/'experiment.yaml';can=yaml.safe_load(path.read_text())
        assert can['status']=='completed_audited_precision_qualified_base_only' and not can['active_processes']
        result=root/can['result_directory']/'summary.json';precision=json.loads(result.read_text())
        assert precision['status']=='PASS_CLEAN_BASE_CPU_GPU_FIXTURE' and not precision['failures']
        assert precision['original_source_audit_unchanged'] and precision['cpu_precision']=='float64'
        for src in (path,result):bound[str(src.relative_to(root))]=sha256(src.read_bytes()).hexdigest()
    if cfg.get('utility_source'):
        path=root/'experiments'/cfg['utility_source']/'experiment.yaml';can=yaml.safe_load(path.read_text())
        assert not can.get('active_processes') and can['aggregation_exit_code']==0
        assert all(j['status']=='completed_audited' for j in can['jobs'].values())
        result=root/can['result_directory']/'summary.json';utility=json.loads(result.read_text())
        assert not utility['unavailable']
        for src in (path,result):bound[str(src.relative_to(root))]=sha256(src.read_bytes()).hexdigest()
    for eid in cfg.get('correction_sources',[]):
        path=root/'experiments'/eid/'experiment.yaml';can=yaml.safe_load(path.read_text());cc=json.loads((root/can['config']).read_text())
        bound[str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
        directory=cc.get('reaudit_directory',cc.get('output_directory'));assert directory
        bound[can['config']]=sha256((root/can['config']).read_bytes()).hexdigest()
        for jid,state in can.get('jobs',{}).items():corrections[jid]=(eid,state,root/directory/jid/'audit.json')
    for eid in source_ids:
        path=root/'experiments'/eid/'experiment.yaml';can=yaml.safe_load(path.read_text())
        bound[str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
        source_cfg=json.loads((root/can['config']).read_text());bound[can['config']]=sha256((root/can['config']).read_bytes()).hexdigest()
        for job in source_cfg['jobs']:
            if selected[slot(job)]!=(eid,job['id']):continue
            jid=job['id'];kind=job['kind'];dim=job.get('after_dimension',job.get('dimension',768 if kind=='EXBert' else 200))
            key=(job['method'],kind,dim);state=can.get('jobs',{}).get(jid,dict(status='queued'))
            status=state['status'];entry=dict(id=jid,seed=job['seed'],status=status,stage=state.get('stage'),attempt_history=attempts[slot(job)])
            out=root/'artifacts/incremental'/jid
            audit_path=out/'audit.json'
            if jid in corrections:
                ceid,cs,cp=corrections[jid];entry.update(original_status=status,correction_source=ceid,correction_status=cs['status'])
                if cs['status']=='completed_audited':
                    status='completed_audited';entry['status']=status;audit_path=cp
            if status=='completed_audited':
                row=json.loads((out/'evaluation.json').read_text());tr=json.loads((out/'training-receipt.json').read_text());audit=json.loads(audit_path.read_text())
                assert not audit.get('failures') and audit['status'].startswith('PASS'),jid
                assert not tr['test_read'] and tr['checkpoint_sha256']==sha256((out/'best.pt').read_bytes()).hexdigest(),jid
                if audit['status']=='PASS_COMMON_LM_NATIVE_GPU_DELTA_CPU_RANK_AUDIT':
                    assert audit['source_checkpoint_sha256']==tr['checkpoint_sha256'] and not audit['retrained']
                    assert audit['source_original_audit_sha256']==sha256((out/'audit.json').read_bytes()).hexdigest()
                    assert audit['source_training_receipt_sha256']==sha256((out/'training-receipt.json').read_bytes()).hexdigest()
                    assert audit['max_native_delta_error']<=audit['original_native_delta_tolerance']==1e-5
                    assert audit['rank_mismatches']==0 and audit['candidate_score_checks']==1080768
                    entry['audit_qualification']=dict(status=audit['status'],native_delta_replay_device=audit['native_delta_replay_device'],
                        native_steps_replayed=audit['native_steps_replayed'],max_native_delta_error=audit['max_native_delta_error'],
                        cpu_max_abs_logit_error=audit['max_abs_logit_error'],cpu_max_initial_logit_error=audit['max_initial_logit_error'],
                        original_audit_status=audit['source_original_audit']['status'],original_audit_unchanged=True,limits=audit['limits'])
                assert row['candidates']==cfg['candidates'] and row['old_after']['queries']==cfg['old_queries'] and row['new_after']['queries']==cfg['new_queries']
                assert row['seed']==job['seed'] and row['method']==job['method'] and row['kind']==kind
                assert row.get('dimension',dim)==dim
                if 'Rank' in job['method'] or 'Novel' in job['method'] or job['method'].startswith('LearnedProposal'):
                    assert audit.get('solver_audit',{}).get('status')=='PASS_ALL_STEPS_SHARED_MODEL_AND_QP_SYSTEMS'
                required_solver=cfg.get('solver_audit_status_by_method',{}).get(job['method'])
                if required_solver:
                    solver=audit.get('solver_audit',{});assert solver.get('status')==required_solver and not solver.get('failures')
                    assert solver['patches_replayed']==3753 and solver['terminal_cpu_counts_exact']
                    assert solver['train_only_parameter_selection_rebuilt'] and solver['weights_and_radii_independently_checked']
                with np.load(out/'ranks.npz') as raw:
                    before=raw['old_before'];after=raw['old_after'];good=before<=10
                    row['old_top10_lost']=int((after[good]>10).sum())
                    row['old_rank_displacement_p95']=float(np.percentile(after-before,95))
                    row['old_query_sha256']=sha256(raw['old_queries'].tobytes()).hexdigest()
                    row['new_query_sha256']=sha256(raw['new_queries'].tobytes()).hexdigest()
                    assert int(good.sum())==row['old_top10_protected'] and row['old_top10_loss']==float((after[good]>10).mean())
                    nq=raw['new_queries'];nr=raw['new_after'];new_h=nq[:,0]>=7505;new_t=nq[:,2]>=7505
                    cohorts=dict(query_head_new_gold_old=new_h&~new_t,query_head_old_gold_new=~new_h&new_t,both_new=new_h&new_t)
                    assert sum(int(mask.sum()) for mask in cohorts.values())==cfg['new_queries']
                    row['new_query_cohorts']={name:dict(queries=int(mask.sum()),raw_mrr=float((1/nr[mask]).mean()),hits10=float((nr[mask]<=10).mean())) for name,mask in cohorts.items()}
                row['training']=tr;groups[key].append(row);checks+=8
                for name in ['evaluation.json','training-receipt.json','audit.json']:
                    src=audit_path if name=='audit.json' else out/name
                    bound[str(src.relative_to(root))]=sha256(src.read_bytes()).hexdigest()
            elif status in ['failed_retained','blocked_by_retained_method_preflight_failure']:
                if (out/'audit.json').exists():entry['failure_evidence']=json.loads((out/'audit.json').read_text()).get('failures',[])
                entry['exit_code']=state.get('exit_code')
            states[key].append(entry)
    records=[]
    for key,ss in states.items():
        method,kind,dim=key;rows=sorted(groups[key],key=lambda x:x['seed']);seeds=[r['seed'] for r in rows]
        assert len(seeds)==len(set(seeds)) and set(seeds)<=set(cfg['seeds'])
        metrics={}
        values=dict(old_mrr=[r['old_after']['raw_mrr'] for r in rows],new_mrr=[r['new_after']['raw_mrr'] for r in rows],
                    old_top10_loss=[r['old_top10_loss'] for r in rows],same_pool_loss=[r['same_pool_top10_loss'] for r in rows],
                    old_rank_displacement_p95=[r['old_rank_displacement_p95'] for r in rows],
                    new_hits10=[r['new_after']['hits10'] for r in rows],displacement=[r['mean_rank_displacement'] for r in rows],
                    incremental_training_seconds=[r['training'].get('source_inclusive_incremental_elapsed_s',r['training']['elapsed_s']+r['training'].get('source_sgd_elapsed_s',0.)) for r in rows])
        for name,v in values.items():
            if v:metrics[name]=dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else None,per_seed=v)
        records.append(dict(method=method,model=kind,dimension=dim,audited_seeds=seeds,complete_seed_set=seeds==cfg['seeds'],metrics=metrics,
                            per_seed_status=sorted(ss,key=lambda s:s['seed']),query_checks=[dict(seed=r['seed'],old_query_sha256=r['old_query_sha256'],new_query_sha256=r['new_query_sha256'],
                                old_top10_n=r['old_top10_protected'],old_top10_lost=r['old_top10_lost']) for r in rows],
                            new_directional_metrics=[dict(seed=r['seed'],cohorts=r['new_query_cohorts']) for r in rows],
                            training_qualifications=[dict(seed=r['seed'],limits=r['training'].get('limits',[]),parameters=r['training'].get('parameters'),
                                trainable_parameters=r['training'].get('trainable_parameters'),peak_gpu_memory_bytes=r['training'].get('peak_gpu_memory_bytes'),
                                meta_elapsed_s=r['training'].get('meta_elapsed_s'),base_elapsed_s=r['training'].get('base_elapsed_s'),source_sgd_elapsed_s=r['training'].get('source_sgd_elapsed_s'),
                                scope_preparation_elapsed_s=r['training'].get('scope_preparation_elapsed_s'),modifiable_parameters=r['training'].get('modifiable_parameters'),
                                guard_scope=r['training'].get('guard_scope'),guard_arithmetic=r['training'].get('guard_arithmetic'),
                                selected_old_rows=len(r['training']['selected_old_rows']) if 'selected_old_rows' in r['training'] else None,
                                processed_new_entities=r['training'].get('processed_new_entities'),accepted_updates=r['training'].get('accepted_updates'),
                                unchanged_fallback_updates=r['training'].get('zero_fallbacks'),steps_modifying_old_rows=r['training'].get('steps_modifying_old_rows'),
                                terminal_guard_recount=r['training'].get('terminal_complete_guard_recount'),
                                postprocessing_elapsed_s=r['training']['elapsed_s'] if 'source_sgd_elapsed_s' in r['training'] else None) for r in rows]))
    order=['LKGE','IncDE','FastKGE','STARK','SAGE','KGEditor','RandomExpansion','NewEntityOnly','FullFinetune','Replay']
    records.sort(key=lambda r:(order.index(r['method']) if r['method'] in order else len(order),r['method'],r['model']))
    directional=[]
    by_method={r['method']:r for r in records if r['model']=='DistMult'}
    if all(m in by_method and by_method[m]['complete_seed_set'] for m in ('LearnedProposal','LearnedProposalQPPred')):
        source=by_method['LearnedProposal']['new_directional_metrics'];after=by_method['LearnedProposalQPPred']['new_directional_metrics']
        for cohort in source[0]['cohorts']:
            for metric in ('raw_mrr','hits10'):
                a=[r['cohorts'][cohort][metric] for r in source];b=[r['cohorts'][cohort][metric] for r in after]
                assert [r['seed'] for r in source]==[r['seed'] for r in after]==cfg['seeds']
                assert all(x['cohorts'][cohort]['queries']==y['cohorts'][cohort]['queries'] for x,y in zip(source,after))
                directional.append(dict(cohort=cohort,metric=metric,queries_per_seed=source[0]['cohorts'][cohort]['queries'],
                                        source_mean=float(np.mean(a)),qp_mean=float(np.mean(b)),retention_ratio=float(np.mean(b)/np.mean(a)),
                                        paired_seed_deltas=[y-x for x,y in zip(a,b)]))
    report=dict(scope='Audited exposed development comparison; pending and failed seeds remain explicit',candidate_count=cfg['candidates'],old_queries=cfg['old_queries'],new_queries=cfg['new_queries'],
                completed_jobs=sum(len(v) for v in groups.values()),planned_jobs=sum(len(v) for v in states.values()),assembly_checks=checks,source_sha256=bound,rows=records,development_utility_screen=utility,additional_development_utility=additional_utility,learned_qp_directional_tradeoff=directional,foundation_precision_adjudication=precision,protection_scope_diagnostic=scope_diagnostic,shared_guard_initial_coverage=shared_scope)
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=True);(out/'table.json').write_text(json.dumps(report,indent=2))
    lines=['# 新增实体共同任务：已审计结果与失败状态','',f"{report['completed_jobs']}/{report['planned_jobs']}个注册方法/种子任务通过审计。运行、排队与失败均逐seed记录；失败不填成绩。",'',
           'after均为11258候选，旧TEST74428／新TEST12406方向查询，rank=1+count(score>gold)，无过滤。当前是已暴露的开发比较。','',
           '| 方法 | 模型/维数 | 审计seed数 | 旧MRR | 新MRR | 旧Top10掉榜% | 新Hits10% | 增量秒 | 状态17/29/43 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    labels=dict(completed_audited='通过',failed_retained='失败留档',running='运行',waiting_audit='待审计',queued='排队',blocked_by_retained_method_preflight_failure='预检失败阻断')
    for row in records:
        m=row['metrics'];state='/'.join(labels.get(x['status'],x['status']) for x in row['per_seed_status'])
        values=[f"{m[k]['mean']*scale:.5f}" if k in m else '—' for k,scale in [('old_mrr',1),('new_mrr',1),('old_top10_loss',100),('new_hits10',100),('incremental_training_seconds',1)]]
        lines.append(f"| {row['method']} | {row['model']}/{row['dimension']} | {len(row['audited_seeds'])}/3 | "+' | '.join(values)+f' | {state} |')
    lines+=['','数值仅取已通过审计seed的均值；不足3seed的行不能作为完整三seed比较。各seed状态、标准差、同池掉榜、位移与阶段成本在table.json。旧掉榜分母为原7505池Top10成功查询，同池辅助分母为11258初始化池成功查询。','',
            'Rank／QP采用同值初始化的float64共享实体连续更新，单轮新TRAIN扫遍；基线SGD有不同轮次和成本，不能据此宣布等成本胜利。Bounded是公开的稳定性适配，半径只依赖旧底座及随机扩展初态；它不是原论文公式或已确认创新。','',
            '原始Rank、Ridge的数值不稳定及QP审计错误保留；单列原始与稳定版本，不能将共同共享实体适配失败外推为原论文独立编辑失败。QP被接受时只保护注册TRAIN集合的当前边界；初始化损失、回退及未保护TEST无硬保证。','',
            'KGEditor使用干净语言模型底座、原生元编辑器和累积在线编辑，另计底座／meta成本，描述文本及预训练知识属于额外信息。尚未通过完整审计时不填最终成绩。','',
            'LearnedProposal各行复用已审计NewEntityOnly训练权重，以固定三seed消融SVD、TRUE／Prediction与最近投影。增量秒包含复用SGD阶段加投影阶段，JSON另列阶段耗时；复用不等于新重训，也不是新算法确认。所有候选仍需真实旧／新排名和预设新知识效用门槛。JSON还按“新查询头／旧答案、旧查询头／新答案、两端均新”拆分新知识效果，保留全部12406查询。','',
            'FastKGE／STARK旧E/R不变仍可能旧答案掉榜，支持参数保持不足以保证新增候选竞争下的排名保持；不能概括所有旧方法仅保护分数。LearnedProposalQPPred相对同底座NewEntityOnly降低旧Top10损失，同时牺牲少量新MRR；不支持全面优于所有文献基线或原论文的结论。','',
            '实现边界：STARK消费距离列；IncDE排序文件被作者loader覆盖、degree分支复用betweenness；SAGE旧producer元数据FAIL仍保留。完整来源适配及审计限制沿用已注册协议。审计核查全查询身份/聚合与固定完整候选评分行，不等于独立重训所有优化器。','']
    if utility:
        lines+=['## 独立VALID效用复算','',
                '以下是固定开发集门槛，非最终留出认证：三seed平均新MRR至少保留98%，平均旧Top10损失最多为源模型75%，且每seed旧损失不恶化。18个任务均复算事前固定的旧／新VALID各1024条查询及完整候选池，不是整个VALID split；审计通过不等于科学收益门槛通过。','',
                '| 方案 | 固定开发门槛 |','|---|---|']
        lines += [f"| {r['method']} | {r['decision']} |" for r in utility['rows']]
    if directional:
        labels={'query_head_new_gold_old':'新头实体查询／旧答案','query_head_old_gold_new':'旧头实体查询／新答案','both_new':'两端均新'}
        lines+=['','## 新知识检索方向的代价','',
                '总体新MRR保留约99%不能代表每个方向都保留99%。以下按同一批完整新TEST查询拆分；这些已暴露开发结果不作为未见留出认证，也不回写已冻结的VALID门槛。','',
                '| 检索方向 | 指标 | 无QP | TRUE＋Prediction QP | 保留比例 |','|---|---|---:|---:|---:|']
        lines += [f"| {labels[r['cohort']]} | {r['metric']} | {r['source_mean']:.5f} | {r['qp_mean']:.5f} | {100*r['retention_ratio']:.2f}% |" for r in directional]
        lines+=['','旧头实体查询／新答案方向MRR下降约10%、Hits10下降约20%；当前QP方案仍存在新增答案检索代价。该方向是后续优化需要解决的问题，不能仅据总体VALID门槛通过就宣称新知识写入已无明显代价。']
    if precision:
        lines+=['','KGEditor17号底座原float32 CPU排名审计失败留档：一个近同分候选使rank240/241不一致。独立注册的016复审采用CPU float64，64查询/480320分数在原1e-4容差内且排名全相同；GPU float32重新加载分数误差0。017用同一权重的独立底座目录继续，后续元训练／在线／最终审计门槛不变。008失败及所有继续尝试可在JSON的attempt_history核对。']
    if any(s.get('audit_qualification') for r in records for s in r['per_seed_status']):
        lines+=['','KGEditor三个种子的031复核统一采用两步原生GPU参数增量重放（原1e-5容差，误差0），另独立完成32条初态和64条终态查询的完整候选CPU评分核查（原1e-4容差、严格排名完全一致）及全查询身份/指标聚合核查。不是全训练过程独立重训或全部查询CPU重新推理。43号原CPU增量重放误差1.63e-5超过1e-5的失败记录保持不变。三seed在本次累计新增实体适配中新MRR约0.00095、旧Top10损失约99.93%；该适配的失败不代表作者原单次编辑协议失败。逐seed资格和原审计状态见JSON。']
    for screen in additional_utility:
        description=cfg.get('utility_descriptions',{}).get(screen['experiment'],{})
        lines+=['','## '+description.get('title','任务方向投影的固定验证'),'',description.get('text',
                '019比较欧氏与TRAIN新答案评分方向加权的TRUE＋Prediction投影，各三seed。020独立复算要求整体新MRR至少保留98%、旧掉榜最多为源模型75%、每seed旧损伤不恶化，并要求新答案检索方向MRR至少保留98%、Hits10至少保留95%；加权方案还须不劣于欧氏控制。未完成时不能认定优化有效。'),'']
        if screen['result']:
            lines+=['| 方法 | 固定方向验证结论 |','|---|---|']+[f"| {r['method']} | {r['decision']} |" for r in screen['result']['rows']]
        else:lines.append(f"独立验证尚无终态结论：{screen['status']}。")
    if scope_diagnostic:
        mean_fraction=float(np.mean([s['union_fraction'] for s in scope_diagnostic['seeds']]))
        lines+=['','## TRAIN保护范围诊断','',
                f'021在16757条旧头／新答案TRAIN事实中检查到三seed平均{100*mean_fraction:.2f}%落在初态保持的原第10名边界上。冻结旧E/R且保持严格边界时，这些新答案仍至少有10个旧候选分数更高；两类端点、三seed全部旧候选独立评分均验证rank下界11。该结果只解释一个充分的TRAIN阻碍子集，不是VALID/TEST全损伤因果归因。',
                '', '固定加权候选未通过020门槛，保留负结果；不扩大系数或挑选seed。共享向量构造例显示，调整一个旧竞争候选可以保住旧真答案第10名、让新答案进入前10，同时保持未受影响关系的预测集合；这只是构造可行性，真实旧参数改动仍须重算所有共享head/tail保护查询。细节见plan/review/task-projection-disposition.md。']
    if shared_scope:
        lines+=['','## 共享更新保护合同的初态覆盖','',
                '024先在扩展至11258候选的初始化模型上筛选仍处于Top10的原参考集合。硬保护只涵盖这些初态合格成员；TRUE参考沿用既有TRAIN选择上限。限定保护还豁免由新TRAIN确定的受影响查询的Prediction成员，并保留其合格TRUE成员。全部旧／新TEST查询仍参与评价。以下是准备阶段覆盖，不是完整方法的收益。','',
                '| seed | 选定TRUE／初态不合格 | 原P查询／完整集合初态合格 | TRAIN受影响查询 | 全保护成员／查询 | 限定保护成员／查询 |',
                '|---|---:|---:|---:|---:|---:|']
        lines += [f"| {r['seed']} | {r['source_true_rows']}/{r['true_initially_ineligible']} | {r['source_prediction_queries']}/{r['prediction_initially_retained_queries']} | {r['affected_old_train_query_keys']} | {r['full_reference_members']}/{r['full_grouped_queries']} | {r['scoped_reference_members']}/{r['scoped_grouped_queries']} |" for r in shared_scope]
        lines+=['','共享更新的主表增量秒包含源SGD、024保护范围准备和本次后处理；参数范围、阶段成本、接受数、原样回退数及旧行变化步数在training_qualifications逐seed披露。固定FP64树保护评分与最终共同GEMM评分分别注明。完整终态CPU保护审计不能表述为每一步完整CPU重评分。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps({k:report[k] for k in ['completed_jobs','planned_jobs','assembly_checks']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',default='configs/common-table-v2.json');a=p.parse_args();build(a.root.resolve(),a.config)
