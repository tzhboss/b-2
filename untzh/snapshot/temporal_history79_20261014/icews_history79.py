"""Read archived histories and existing checkpoints, measure prediction validity transitions.

No retraining. Unrecorded predictions are UNKNOWN in the real world, even when
the benchmark calls them incorrect. Preserve this distinction in every output.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/lc/untzh/temporal_drift76_20261014')
sys.path.insert(0,str(SOURCE))
from train76 import setup
from evaluate76 import load_model
from prepare76 import write_json

ARMS=['TDistMult_finetune','TDistMult_replay','TComplEx_finetune','TComplEx_replay','TComplEx_frozen_history']


@torch.no_grad()
def prediction(model,data,ids,n):
    pred=np.empty(len(ids),dtype=np.int64)
    score=np.empty(len(ids),dtype=np.float64)
    ties=np.empty(len(ids),dtype=np.int32)
    for b in range(0,len(ids),256):
        q=torch.as_tensor(data['test'][ids[b:b+256]],device='cuda')
        logits=model.query(q)@model.E[:n].T
        value,index=logits.max(1)
        pred[b:b+len(q)]=index.cpu().numpy()
        score[b:b+len(q)]=value.cpu().numpy()
        ties[b:b+len(q)]=(logits==value[:,None]).sum(1).cpu().numpy()
    return pred,score,ties


def annotate(pred,ids,data,bounds):
    # Every list is exactly the same query entity/relation/date, all standard splits.
    return np.array([int(p) in data['filter_cols'][bounds[idx]:bounds[idx+1]] for p,idx in zip(pred,ids)])


def counts(mask,before_true,after_true,pbefore,pafter,before_ties,after_ties):
    n=int(mask.sum())
    def count(m):return int((mask&m).sum())
    values={'recorded_true_to_recorded_true':count(before_true&after_true),
            'recorded_true_to_unrecorded':count(before_true&~after_true),
            'unrecorded_to_recorded_true':count(~before_true&after_true),
            'unrecorded_to_unrecorded':count(~before_true&~after_true)}
    assert sum(values.values())==n
    changed=mask&(pbefore!=pafter)
    return {'n':n,'before_recorded_true_top1':count(before_true),'after_recorded_true_top1':count(after_true),
            'transitions':values,'transition_rates':{k:v/n if n else None for k,v in values.items()},
            'prediction_entity_changed_n':int(changed.sum()),
            'changed_but_both_recorded_true_n':int((changed&before_true&after_true).sum()),
            'top1_tie_before_n':count(before_ties>1),'top1_tie_after_n':count(after_ties>1),
            'semantic_real_world_error_status':'unrecorded != proven false; no point estimate of true-world error fraction'}


def named_case(i,ids,data,mapping,pbefore,pafter,origin_rank,final_rank,before_true,after_true):
    idx=int(ids[i]);h,r,t,d=map(int,data['test'][idx]);nr=len(mapping['relations'])
    return {'test_directional_index':idx,'query_entity':mapping['entities'][h],
            'relation':mapping['relations'][r%nr],'direction':'tail' if r<nr else 'head',
            'fixed_query_date':mapping['dates'][d],'gold':mapping['entities'][t],
            'original_filtered_gold_rank':float(origin_rank[i]),'final_filtered_gold_rank':float(final_rank[i]),
            'original_top1':mapping['entities'][int(pbefore[i])],
            'final_top1':mapping['entities'][int(pafter[i])],
            'original_top1_recorded_true':bool(before_true[i]),'final_top1_recorded_true':bool(after_true[i]),
            'final_prediction_train_first_year':int(data['eyear'][pafter[i]]),
            'world_judgment':'unknown until exact dated event evidence or explicit contradiction is available'}


def main():
    setup();started=time.perf_counter()
    datafile=np.load(SOURCE/'prepared/data.npz');data={k:datafile[k] for k in datafile.files}
    stats=json.loads((SOURCE/'data-statistics.json').read_text())
    mapping=json.loads((SOURCE/'prepared/mapping.json').read_text())
    ids=np.flatnonzero(data['eligible']&(data['test_year']<2015))
    bounds=np.searchsorted(data['filter_rows'],np.arange(len(data['test'])+1))
    output={'source_experiment':str(SOURCE),'date_semantics':'exact historical date unchanged',
            'primary_population':'all eligible 2005-2014 directional TEST queries, before at own original year and after at 2015',
            'n':len(ids),'historical_truth_expiration_annotations':0,
            'gold_validity_world_correction_evidence':'no record retraction/correction history supplied by ICEWS benchmark',
            'real_world_error_fraction':None,'reasonable_future_world_change_explanation':'cannot expire a fixed earlier event solely because later events differ',
            'arms':{}}
    (ROOT/'icews').mkdir(exist_ok=True)
    base_predictions={}
    for arm in ARMS:
        architecture=arm.split('_')[0]
        before=np.empty(len(ids),dtype=np.int64);before_score=np.empty(len(ids));before_ties=np.empty(len(ids),dtype=np.int32)
        original_rank=np.empty(len(ids));original_raw=np.empty(len(ids))
        for year in range(2005,2015):
            select=data['test_year'][ids]==year;year_ids=ids[select]
            cachekey=(architecture,year) if year==2005 else (arm,year)
            if cachekey in base_predictions:
                bp,bs,bt=base_predictions[cachekey]
            else:
                model,_=load_model(architecture,arm,year,stats)
                n=int((data['eyear']<=year).sum())
                bp,bs,bt=prediction(model,data,year_ids,n)
                base_predictions[cachekey]=(bp,bs,bt)
            before[select]=bp;before_score[select]=bs;before_ties[select]=bt
            old=np.load(SOURCE/f'evaluation/{arm}-{year}.npz')
            slots=np.searchsorted(old['ids'],year_ids)
            assert np.array_equal(old['ids'][slots],year_ids)
            original_rank[select]=old['filtered_full'][slots];original_raw[select]=old['raw_full'][slots]
        model,_=load_model(architecture,arm,2015,stats)
        after,after_score,after_ties=prediction(model,data,ids,int((data['eyear']<=2015).sum()))
        bf=annotate(before,ids,data,bounds);af=annotate(after,ids,data,bounds)
        last=np.load(SOURCE/f'evaluation/{arm}-2015.npz');slots=np.searchsorted(last['ids'],ids)
        assert np.array_equal(last['ids'][slots],ids)
        final_rank=last['filtered_full'][slots];final_raw=last['raw_full'][slots]
        masks={'all_historical':np.ones(len(ids),dtype=bool),'all_rank_worsening':final_rank>original_rank,
               'all_rank_changes':final_rank!=original_rank,'all_original_top10_drops':(original_rank<=10)&(final_rank>10),
               'anchor2005':data['test_year'][ids]==2005,
               'anchor2005_rank_worsening':(data['test_year'][ids]==2005)&(final_rank>original_rank),
               'anchor2005_original_top10_drops':(data['test_year'][ids]==2005)&(original_rank<=10)&(final_rank>10)}
        report={key:counts(mask,bf,af,before,after,before_ties,after_ties) for key,mask in masks.items()}
        output['arms'][arm]=report
        np.savez_compressed(ROOT/f'icews/{arm}-history.npz',ids=ids,prediction_before=before,prediction_after=after,
                            top1_before_score=before_score,top1_after_score=after_score,
                            before_top1_recorded_true=bf,after_top1_recorded_true=af,
                            before_top1_ties=before_ties,after_top1_ties=after_ties,
                            original_rank=original_rank,final_rank=final_rank,original_raw_rank=original_raw,final_raw_rank=final_raw)
        cases=[]
        for label,select in [('benchmark_new_top1_error',masks['anchor2005_rank_worsening']&bf&~af),
                             ('still_recorded_true',masks['anchor2005_rank_worsening']&bf&af),
                             ('baseline_already_unrecorded',masks['anchor2005_rank_worsening']&~bf&~af)]:
            choices=np.flatnonzero(select)
            # Fixed seed, not top-loss-only examples; descriptions are not population error estimates.
            rng=np.random.default_rng(17)
            chosen=rng.choice(choices,min(8,len(choices)),replace=False)
            for i in sorted(chosen):
                cases.append({'selection_group':label,**named_case(i,ids,data,mapping,before,after,original_rank,final_rank,bf,af)})
        write_json(ROOT/f'icews/{arm}-history-examples.json',cases)
        write_json(ROOT/'icews-history-summary.json',output)
        write_json(ROOT/'status.json',{'stage':'history_annotation','completed_arm':arm,'pid':os.getpid()})
        print('HISTORY_COMPLETE',arm,json.dumps(report['anchor2005_rank_worsening']),flush=True)
    output['history_extraction_seconds']=time.perf_counter()-started
    write_json(ROOT/'icews-history-summary.json',output)
    write_json(ROOT/'status.json',{'stage':'icews_history_complete','pid':os.getpid(),'seconds':output['history_extraction_seconds']})
    print('ALL_HISTORY_COMPLETE',output['history_extraction_seconds'],flush=True)


if __name__=='__main__':main()
