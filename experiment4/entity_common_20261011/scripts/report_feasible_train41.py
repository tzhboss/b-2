"""Unchanged original gates after all six complete independent audits."""
import json
from pathlib import Path
import numpy as np
import yaml
import hashlib
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'configs/feasible-train41.json').read_text());state=yaml.safe_load((root/'experiments'/cfg['id']/'experiment.yaml').read_text())
assert state['status']==cfg['terminal_status'] and not state['active_processes']
for name,digest in state['frozen_input_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,digest in state['external_source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert all(j['status']=='completed_audited' for j in state['jobs'].values())
dest=root/'results'/cfg['id'];summary=json.loads((dest/'summary.json').read_text());assert summary['completed_audited_jobs']==6
records=[]
for method in ('FeasibleTrainFull','FeasibleTrainScoped'):
    rows=sorted([r for r in summary['rows'] if r['job']['method']==method],key=lambda x:x['job']['seed'])
    assert [r['job']['seed'] for r in rows]==[17,29,43]
    val=[r['utility']['validation'] for r in rows]
    for r in rows:
        assert not r['audit']['failures'] and r['audit']['train_audit']['terminal_cpu_counts_exact']
        assert r['audit']['train_audit']['patches_replayed']==3753 and r['receipt']['validation']==r['utility']['validation']
    cohort='query_head_old_gold_new'
    sn=np.array([v['source'][1]['raw_mrr'] for v in val]);an=np.array([v['after'][1]['raw_mrr'] for v in val])
    sl=np.array([v['source_old_top10_loss'] for v in val]);al=np.array([v['after_old_top10_loss'] for v in val])
    sm=np.array([v['directional'][cohort]['source']['raw_mrr'] for v in val]);am=np.array([v['directional'][cohort]['after']['raw_mrr'] for v in val])
    sh=np.array([v['directional'][cohort]['source']['hits10'] for v in val]);ah=np.array([v['directional'][cohort]['after']['hits10'] for v in val])
    gates=dict(new_mrr=bool(an.mean()>=.98*sn.mean()),old_loss=bool(sl.mean()>0 and al.mean()<=.75*sl.mean() and np.all(al<=sl)),
               direction=bool(am.mean()>=.98*sm.mean() and ah.mean()>=.95*sh.mean()))
    paired=[]
    for r in rows:
        original=root/'artifacts/incremental'/r['job']['initial_endpoint']
        parent=json.loads((original/'training-receipt.json').read_text())['validation'];pe=json.loads((original/'evaluation.json').read_text())
        now=r['utility']['validation'];ev=r['evaluation']
        paired.append(dict(seed=r['job']['seed'],new_valid_mrr_delta=now['after'][1]['raw_mrr']-parent['after'][1]['raw_mrr'],
             directional_mrr_delta=now['directional'][cohort]['after']['raw_mrr']-parent['directional'][cohort]['after']['raw_mrr'],
             old_valid_loss_delta=now['after_old_top10_loss']-parent['after_old_top10_loss'],
             new_test_mrr_delta=ev['new_after']['raw_mrr']-pe['new_after']['raw_mrr'],old_test_loss_delta=ev['old_top10_loss']-pe['old_top10_loss']))
    def values(arr):return dict(values=arr.tolist(),mean=float(arr.mean()),std=float(arr.std(ddof=1)))
    records.append(dict(method=method,decision='PASS_DIRECTIONAL_DEVELOPMENT_SCREEN' if all(gates.values()) else 'NO_GO_FOR_THIS_FIXED_COMBINATION',gates=gates,
        new_valid_mrr=values(an),old_valid_loss=values(al),directional_mrr=values(am),directional_hits10=values(ah),
        source_new_valid_mrr=values(sn),source_old_valid_loss=values(sl),source_directional_mrr=values(sm),source_directional_hits10=values(sh),
        full_test_new_mrr=values(np.array([r['evaluation']['new_after']['raw_mrr'] for r in rows])),full_test_old_loss=values(np.array([r['evaluation']['old_top10_loss'] for r in rows])),
        directional_mrr_retention=float(am.mean()/sm.mean()),directional_hits10_retention=float(ah.mean()/sh.mean()),paired_original_qp=paired,
        source_inclusive_training_seconds=values(np.array([r['receipt']['source_inclusive_incremental_elapsed_s'] for r in rows])),
        independent_audit_seconds=values(np.array([r['audit']['train_audit']['elapsed_s'] for r in rows]))))
assert not (dest/'utility-disclosure.json').exists()
result=dict(status='PASS_COMPLETE_AUDITS_AND_UNCHANGED_GATE_RECONSTRUCTION',rows=records,registered_gates_changed=False,
      limits=['Exposed development; no heldout/SOTA/novelty claim.','Reuse audited SGD/QP source checkpoints, not independent source retraining.',
              'All compiled guards CPU-replayed; no safety assertion outside selected scope.','Full TEST scored on GPU,64 full query score rows independently audited.'])
(dest/'utility-disclosure.json').write_text(json.dumps(result,indent=2))
lines=['# 完整可行域TRAIN优化041','', '完整三种子、六模型通过独立审计后按原门槛汇总。已暴露开发数据，非最终留出或SOTA结论。','',
       '| 方法 | 新VALID MRR | 旧Top10掉榜% | 旧头/新答案MRR | 方向Hits10% | 结论 |','|---|---:|---:|---:|---:|---|']
for r in records:lines.append(f"| {r['method']} | {r['new_valid_mrr']['mean']:.6f} | {100*r['old_valid_loss']['mean']:.4f} | {r['directional_mrr']['mean']:.6f} | {100*r['directional_hits10']['mean']:.4f} | {r['decision']} |")
lines+=['','原Full/Scoped保护成员、半径、旧参数冻结和收益门槛不变。直接TRAIN损失下降不等于检索泛化收益；039固定旧容量限制仍成立。JSON披露每seed、标准差、源SGD与原QP配对、完整TEST及成本。']
(dest/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(result),flush=True)
