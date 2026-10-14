import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/lc/untzh/trajectory_risk144_20261014')
import experiment144 as prediction
previous=prediction.previous
base=prediction.base
ROOT=Path(__file__).resolve().parent

def status(stage,**kw):base.write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),**kw))

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    begin=time.monotonic()
    n,total,_,new,_,truths=prediction.training_assets(17)
    common_old=base.load('/data/lc/untzh/increment87_20261014/cache/train.npz')['old']
    full_queries=base.load('/data/lc/untzh/full_valid102_20261014/queries.npz')['old_valid']
    bundles=[]
    for seed in [43,29]:
        status('prepare',seed=seed)
        tick=time.monotonic()
        src=Path('/data/lc/untzh')/{43:'parent_confirm110_20261014',29:'risk_increment88_20261014'}[seed]
        models=Path(f'/data/lc/untzh/train_init121_20261014/seed{seed}/models/zero')
        out=ROOT/f'seed{seed}'
        if os.environ.get('RESUME_145')=='1' and seed==43:
            assert (out/'forecast-rows.pt').exists()
            data={split:base.load(out/f'{split}-predictions.npz') for split in ['old_train','old_valid']}
            costs=dict(reused_frozen_initial_attempt=True,total_seconds=None,
                note='completed predictions reused; original stage timing not persisted before path failure, unknown not zero')
            bundles.append((seed,models,out,data,costs))
            continue
        initial=torch.load(models/'initial.pt',map_location='cpu',weights_only=True)
        prefix=torch.load(models/'prefix10.pt',map_location='cpu',weights_only=True)
        tape=base.load(src/'cache/learning.npz')
        queries=dict(old_train=common_old,old_valid=full_queries)
        out.mkdir(parents=True,exist_ok=True)
        assert not (out/'forecast-rows.pt').exists(), 'do not silently regenerate'
        forecast,fcost=prediction.forecasts(17,n,initial,prefix,new,tape,out)
        selected_train=np.sort(base.load(src/'cache/cohort.npz')['ids'])
        data={}
        costs={}
        for split in ['old_train','old_valid']:
            status('forecast_score',seed=seed,split=split)
            pred=base.load(f'/data/lc/untzh/admission_risk124_20261014/cache/{seed}-{split}-half-prediction.npz')
            ids=selected_train if split=='old_train' else pred['ids']
            feature,feature_cost=previous.features(initial,prefix,queries[split],pred,truths,n,total,False,ids,out/f'{split}-features.npz')
            data[split],score_cost=prediction.predict(17,n,initial,queries[split],pred,feature,truths,forecast,out/f'{split}-predictions.npz')
            costs[split]=dict(features=feature_cost,scoring=score_cost)
        prep=dict(gradient_forecast=fcost,scoring=costs,total_seconds=time.monotonic()-tick)
        base.write(out/'preparation.json',prep)
        bundles.append((seed,models,out,data,prep))
    base.write(ROOT/'predictions-frozen.json',dict(all_predictions_before_any_endpoint_read=True,seeds=[43,29],
               formula_144_unchanged=True,not_unseen_confirmation=True,no_test=True))
    reports={}
    for seed,models,out,data,costs in bundles:
        status('label',seed=seed)
        ordinary=base.load(f'/data/lc/untzh/train_init121_20261014/seed{seed}/evaluation/zero.npz')
        anchor=base.load(f'/data/lc/untzh/smallinit_protect126_20261014/seed{seed}/zero/evaluation/anchor_null.npz')
        endpoint=torch.load(models/'epoch-20.pt',map_location='cpu',weights_only=True)
        forecast=torch.load(out/'forecast-rows.pt',map_location='cpu',weights_only=True)
        geometry={name:dict(mean_row_norm=float(forecast[name].double().norm(dim=1).mean()),
                   mean_distance_to_actual=float((forecast[name].double()-endpoint['E'][n:]).norm(dim=1).mean()))
                   for name in ['current','one_batch_linear','stationary_mean']}
        cases={split:prediction.evaluate(17,d,ordinary[split],anchor[split]) for split,d in data.items()}
        target=cases['old_valid']['late_new']['predictions']
        mean=target['stationary_mean_rank']
        current=target['current_full_rank']
        signal=bool(mean['ap'] is not None and all(mean['ap']>target[k]['ap'] for k in ['current_full_rank','current_full_gap','one_batch_linear_rank'])
                    and mean['budgets']['0.05']['hits']>=current['budgets']['0.05']['hits'])
        scenario=target['scenario_mean_rank']
        uncertainty=bool(scenario['ap'] is not None and scenario['ap']>mean['ap']
                    and scenario['budgets']['0.05']['hits']>=mean['budgets']['0.05']['hits'])
        report=dict(cases=cases,costs=costs,forecast_geometry=geometry,mean_signal_go=signal,scenario_increment_go=uncertainty)
        reports[str(seed)]=report
        base.write(out/'summary.json',report)
        print('RISK145',seed,'late',cases['old_valid']['late_new']['events'],
              'AP',{k:v['ap'] for k,v in target.items()},'meanGO',signal,'scenarioGO',uncertainty,flush=True)
    mean_go=all(r['mean_signal_go'] for r in reports.values())
    scenario_go=all(r['scenario_increment_go'] for r in reports.values())
    base.write(ROOT/'summary.json',dict(seeds=reports,total_seconds=time.monotonic()-begin,new_training_configs=0,new_parent_trainings=0,
        gradient_batches=44,mean_ranking_decision='GO diagnostic only' if mean_go else 'NO-GO stop stationary ranking formula',
        resumed_initial_frozen_predictions=os.environ.get('RESUME_145')=='1',
        total_seconds_scope='current attempt; original completed seed43 timing unavailable' if os.environ.get('RESUME_145')=='1' else 'complete run',
        scenario_decision='GO diagnostic only' if scenario_go else 'NO-GO no consistent incremental value',
        no_action_trained=True,no_test=True,all_valid_development=True))
    status('complete',seconds=time.monotonic()-begin)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
