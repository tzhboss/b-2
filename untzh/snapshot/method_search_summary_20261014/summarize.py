"""Summarize saved experiment data and descriptive query-cluster uncertainty."""
import collections
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=Path('/data/lc/untzh')
SRC=Path('/data/lc/tzh/experiment4')
RUNS=[('063','search63_20261013',15),('064','search64_20261013',15),('065','benefit65_20261014',4),
      ('066','frontier66_20261014',13),('067','probability67_20261014',12),('068','balance68_20261014',15),
      ('069','calibration69_20261014',22),('070','conditional70_20261014',44),('071','roles71_20261014',14),
      ('072','adaptation72_20261014',15),('073','refine73_20261014',16),('074','coverage74_20261014',12),
      ('075','stratified75_20261014',21),('076','confirm76_20261014',8),('077','normalized77_20261014',12),('078','blend78_20261014',9)]


def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def boot(a,b,queries,mask):
    q=queries[mask];delta=(1/a[mask]-1/b[mask])
    _,group=np.unique(q[:,:2],axis=0,return_inverse=True)
    n=group.max()+1
    total=np.bincount(group,weights=delta,minlength=n);counts=np.bincount(group,minlength=n)
    rng=np.random.default_rng(17);values=[]
    for _ in range(3000):
        chosen=rng.integers(n,size=n);values.append(total[chosen].sum()/counts[chosen].sum())
    return {'mrr_difference':float(delta.mean()),'query_groups':int(n),'answers':len(q),
            'improved':int((delta>0).sum()),'unchanged':int((delta==0).sum()),'worsened':int((delta<0).sum()),
            'descriptive_95_query_bootstrap':np.quantile(values,[.025,.975]).tolist(),
            'limitations':'Post-selection, single-seed exposed VALID; not corrected for247 searches; shared entities create cross-query dependencies.'}


def main():
    started=time.monotonic();reports={};summary={'runs':[]}
    for number,directory,count in RUNS:
        root=BASE/directory;status=read(root/'status.json');cost=read(root/'costs.json')
        if status['stage']!='complete':
            assert number=='064' and 'COMPLETE' in (root/'run.log').read_text() and 'query_old_only' in (root/'extra-train.log').read_text()
            assert 'Traceback' not in (root/'extra-train.log').read_text()
            status={'stage':'completed_from_logs','saved_status':status,
                    'note':'064 supplement finished and saved metrics; its rank helper left status.json at evaluate. No new science or status rewrite.'}
        reports[number]=read(root/'evaluation/metrics.json')
        training=cost.get('training',{})
        stages=sum(v['seconds'] for v in training.values()) if isinstance(training,dict) else 0.
        summary['runs'].append({'round':number,'directory':str(root),'configs':count,'status':status,
            'recorded_total_seconds':cost.get('total_seconds'), 'recorded_training_sum_seconds':stages,
            'note':'072 total_seconds is continuation only, training_sum includes saved original arms; 064/070 supplementary evaluation costs separate.'})
    summary['total_configs']=sum(count for _,_,count in RUNS)
    summary['new_training_configs']=summary['total_configs']-22-44
    summary['intervention_branches']=160
    candidate=reports['077']['cohort_blend0.5']
    baseline=reports['077']['uniform']
    ordinary=reports['077']['new_only']
    summary['candidate']={'name':'cohort_blend0.5','source':str(BASE/'normalized77_20261014'),
       'new_mrr':candidate['new_valid']['all']['filtered_train_truths']['mrr'],
       'old_valid_mrr':candidate['old_valid']['all']['filtered_train_truths']['mrr'],
       'old_train_mrr':candidate['old_train']['all']['filtered_train_truths']['mrr'],
       'old_valid_drops':candidate['old_valid']['drops']['filtered_train_truths']['drops'],
       'old_train_drops':candidate['old_train']['drops']['filtered_train_truths']['drops'],
       'new_relative_gain_vs_uniform':candidate['new_valid']['all']['filtered_train_truths']['mrr']/baseline['new_valid']['all']['filtered_train_truths']['mrr']-1,
       'old_train_drop_reduction_vs_uniform':1-candidate['old_train']['drops']['filtered_train_truths']['drops']/baseline['old_train']['drops']['filtered_train_truths']['drops'],
       'old_train_drop_reduction_vs_ordinary':1-candidate['old_train']['drops']['filtered_train_truths']['drops']/ordinary['old_train']['drops']['filtered_train_truths']['drops']}
    dominance_checks={}
    for split in ['old_train','old_valid','new_valid']:
        kinds=['all'] if split!='new_valid' else ['all','head_old_gold_new','head_new_gold_old','both_new']
        for direction in kinds:
            for metric in ['raw','filtered_train_truths']:
                dominance_checks[f'{split}/{direction}/{metric}']=candidate[split][direction][metric]['mrr']>=baseline[split][direction][metric]['mrr']
    for split in ['old_train','old_valid']:
        for kind in ['raw','filtered_train_truths']:
            assert candidate[split]['drops'][kind]['drops']<=baseline[split]['drops'][kind]['drops']
    summary['candidate']['saved_mrr_dominance_checks']=dominance_checks
    summary['candidate']['saved_metrics_weakly_dominate_uniform_raw_filtered_mrr_and_drop_counts']=all(dominance_checks.values())
    dirs=['head_old_gold_new','head_new_gold_old','both_new']
    summary['candidate']['directional_ratios_vs_ordinary']=[candidate['new_valid'][d]['filtered_train_truths']['mrr']/ordinary['new_valid'][d]['filtered_train_truths']['mrr'] for d in dirs]
    valid=load(SRC/'entity_refresh43_20261012/artifacts/data/validation.npz')
    c=load(BASE/'normalized77_20261014/evaluation/cohort_blend0.5-valid.npz')['new_valid']
    u=load(BASE/'risk58_20261013/evaluation/uniform.npz')['new_valid']
    masks={'all':np.ones(len(c),bool),dirs[0]:(valid['new'][:,0]<7505)&(valid['new'][:,2]>=7505),
           dirs[1]:(valid['new'][:,0]>=7505)&(valid['new'][:,2]<7505),dirs[2]:(valid['new'][:,0]>=7505)&(valid['new'][:,2]>=7505)}
    summary['descriptive_uncertainty']={d:boot(c[:,1],u[:,1],valid['new'],mask) for d,mask in masks.items()}
    controls={name:reports['078'][name] for name in ['onset10_absolute_normmatched','onset10_absolute_075','onset10_mass_normmatched','onset5_blend','onset5_absolute','onset0_blend']}
    summary['controls']=controls
    summary['cost_note']='077 costs include12 trial configurations; deployment requires just one selected host and two sampled protection gradients. All prefix/cache/oldlogZ/calibration historical costs remain necessary, not free.'
    costs=read(BASE/'normalized77_20261014/costs.json')
    summary['candidate_training_seconds']=costs['training']['cohort_blend0.5']['seconds']
    summary['uniform_training_seconds_historical']=read(BASE/'risk58_20261013/costs.json')['training']['uniform']['seconds']
    summary['elapsed_summary_seconds']=time.monotonic()-started
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    rows=['# 063–078 全部配置索引','', '| 轮次 | 配置数 | 训练秒数合计 | 记录总秒数 | 目录 |','|---|---:|---:|---:|---|']
    for row in summary['runs']:
        rows.append(f"| {row['round']} | {row['configs']} | {row['recorded_training_sum_seconds']:.3f} | {row['recorded_total_seconds']} | {row['directory']} |")
    rows+=['','247是配置数，不是247种独立算法；181个训练配置+66个无重训校准配置，另160个37步TRAIN干预分支。',
           '072总时间为接续时间，不计此前12组，不能直接合并当全部实际墙时；064/070追加评价另记。原失败日志保留。',
           '所有失败/未计算TRAIN条目保留在每轮evaluation/metrics.json，不以NA当0。']
    (ROOT/'全部配置索引.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({'candidate':summary['candidate'],'uncertainty':summary['descriptive_uncertainty']['all'],
                      'candidate_training_seconds':summary['candidate_training_seconds'],'configs':summary['total_configs']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
