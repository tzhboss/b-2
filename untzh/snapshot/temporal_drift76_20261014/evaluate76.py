"""One GPU evaluation pass per model/year; exact date filters and rank decomposition."""
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from prepare76 import write_json
from train76 import TemporalBilinear, setup

ROOT=Path(__file__).resolve().parent
ARMS=['TDistMult_finetune','TDistMult_replay','TComplEx_finetune','TComplEx_replay','TComplEx_frozen_history']


def metrics(ranks):
    if not len(ranks):
        return {'n':0,'mrr':None,'hits10':None,'mean_rank':None,'median_rank':None}
    return {'n':len(ranks),'mrr':float(np.mean(1/ranks)),'hits10':float(np.mean(ranks<=10)),
            'mean_rank':float(np.mean(ranks)),'median_rank':float(np.median(ranks))}


def drift(before,after,fixed):
    if not len(before):
        return {'n':0}
    total=after-before;parameter=fixed-before;candidate=after-fixed
    assert np.array_equal(total,parameter+candidate) and (candidate>=0).all()
    worsens=total>0; oldtop=before<=10
    counts={'parameter_only':int((worsens&(parameter>0)&(candidate==0)).sum()),
            'candidate_required_despite_nonpositive_parameter':int((worsens&(parameter<=0)&(candidate>0)).sum()),
            'mixed_positive_parameter_and_candidates':int((worsens&(parameter>0)&(candidate>0)).sum())}
    assert sum(counts.values())==int(worsens.sum())
    return {'n':len(before),'before':metrics(before),'after':metrics(after),'after_fixed_candidates':metrics(fixed),
            'worsened_count':int(worsens.sum()),'worsened_rate':float(worsens.mean()),
            'improved_count':int((total<0).sum()),'unchanged_count':int((total==0).sum()),
            'mean_hrd':float(total.mean()),'median_hrd':float(np.median(total)),
            'hrd_quantiles':dict(zip(['p05','p25','p50','p75','p95'],map(float,np.quantile(total,[.05,.25,.5,.75,.95])))),
            'old_top10_n':int(oldtop.sum()),'old_top10_dropped':int((oldtop&(after>10)).sum()),
            'old_top10_drop_rate':float(np.mean(after[oldtop]>10)) if oldtop.any() else None,
            'entity_interference_count':int((candidate>0).sum()),'entity_interference_rate':float(np.mean(candidate>0)),
            'mean_new_candidate_rank_contribution':float(candidate.mean()),
            'mean_parameter_rank_contribution':float(parameter.mean()),
            'positive_parameter_rank_mass':float(np.maximum(parameter,0).sum()),
            'new_candidate_rank_mass':float(candidate.sum()),'worsening_categories':counts}


def load_model(architecture,arm,year,stats):
    path=ROOT/f'models/{architecture}_shared/2005.pt' if year==2005 else ROOT/f'models/{arm}/{year}.pt'
    saved=torch.load(path,map_location='cuda',weights_only=False)
    model=TemporalBilinear(architecture,stats['entities'],stats['relations'],stats['dates'])
    model.load_state_dict(saved['state']);model.eval()
    return model,saved['metadata']


@torch.no_grad()
def ranks(model,data,ids,active_n,old_n,origin_n,arm,year):
    test=data['test'];fr=data['filter_rows'];fc=data['filter_cols']
    out={k:np.empty(len(ids),dtype=np.float64) for k in
         ('raw_full','raw_previous_candidates','raw_origin_candidates','filtered_full',
          'filtered_previous_candidates','filtered_origin_candidates','gold_score')}
    bounds=np.searchsorted(fr,np.arange(len(test)+1))
    candidate_ids=torch.arange(active_n,device='cuda')[None,:]
    batch_size=256
    started=time.perf_counter()
    for begin in range(0,len(ids),batch_size):
        qids=ids[begin:begin+batch_size]
        x=torch.as_tensor(test[qids],device='cuda')
        score=model.query(x)@model.E[:active_n].T
        target=score[torch.arange(len(x),device='cuda'),x[:,2]].clone()
        assert torch.isfinite(score).all()
        score[torch.arange(len(x),device='cuda'),x[:,2]]=-torch.inf
        origins=torch.as_tensor(origin_n[qids],device='cuda')[:,None]
        for kind in ('raw','filtered'):
            if kind=='filtered':
                local_rows=[];cols=[]
                for i,idx in enumerate(qids):
                    lo,hi=bounds[idx:idx+2]
                    local_rows.extend([i]*(hi-lo));cols.extend(fc[lo:hi])
                if cols:
                    cc=np.array(cols);rr=np.array(local_rows);keep=cc<active_n
                    score[torch.as_tensor(rr[keep],device='cuda'),torch.as_tensor(cc[keep],device='cuda')]=-torch.inf
            # Average rank for exact ties, excluding the true answer itself.
            greater=score>target[:,None];ties=score==target[:,None]
            allr=1+greater.sum(1)+0.5*ties.sum(1)
            prev=1+greater[:,:old_n].sum(1)+0.5*ties[:,:old_n].sum(1)
            mask=candidate_ids<origins
            orig=1+(greater&mask).sum(1)+0.5*(ties&mask).sum(1)
            for name,value in [('full',allr),('previous_candidates',prev),('origin_candidates',orig)]:
                out[kind+'_'+name][begin:begin+len(qids)]=value.cpu().numpy()
        out['gold_score'][begin:begin+len(qids)]=target.cpu().numpy()
        if (begin//batch_size)%40==0:
            write_json(ROOT/'status.json',{'stage':'evaluate','arm':arm,'year':year,'completed_queries':begin+len(qids),
                                          'total_queries':len(ids),'pid':os.getpid()})
            print('EVAL_PROGRESS',arm,year,begin+len(qids),len(ids),flush=True)
    out['ids']=ids
    out['seconds']=np.array(time.perf_counter()-started)
    return out


@torch.no_grad()
def examples(model,data,mapping,ids,origin,final,fixed,max_cases=12):
    total=final-origin;candidate=final-fixed;parameter=fixed-origin
    qualifying=(origin<=10)&(final>10)
    selected=[]
    categories=[('candidate_required',(parameter<=0)&(candidate>0)),
                ('parameter_only',(parameter>0)&(candidate==0)),('mixed',(parameter>0)&(candidate>0))]
    for label,mask in categories:
        local=np.flatnonzero(qualifying&mask)
        chosen=local[np.argsort(-total[local],kind='stable')[:max_cases//3]]
        selected.extend((label,int(j)) for j in chosen)
    nr=len(mapping['relations']);truth={}
    for row in np.concatenate([data['train'],data['valid'],data['test']]):
        h,r,t,d=map(int,row);truth.setdefault((h,r,d),set()).add(t)
    cases=[]
    n=int((data['eyear']<=2015).sum())
    for label,j in selected:
        idx=int(ids[j]);h,r,t,d=map(int,data['test'][idx]);x=torch.as_tensor(data['test'][idx:idx+1],device='cuda')
        scores=(model.query(x)@model.E[:n].T)[0]
        for known in truth[(h,r,d)]:
            if known<n:scores[known]=-torch.inf
        vals,top=torch.topk(scores,10)
        competitors=[{'entity':mapping['entities'][int(c)],'train_first_year':int(data['eyear'][int(c)]),
                      'score':float(v),'recorded_true_for_this_exact_query_date':int(c) in truth[(h,r,d)]}
                     for c,v in zip(top,vals)]
        cases.append({'category':label,'test_directional_index':idx,'direction':'tail' if r<nr else 'head',
                      'query_entity':mapping['entities'][h],'relation':mapping['relations'][r%nr],
                      'date':mapping['dates'][d],'gold':mapping['entities'][t],
                      'original_rank':float(origin[j]),'final_rank':float(final[j]),
                      'final_rank_original_candidates':float(fixed[j]),'parameter_contribution':float(parameter[j]),
                      'new_candidate_contribution':float(candidate[j]),'top_filtered_competitors':competitors})
    return cases


def main():
    setup();start=time.perf_counter()
    assert json.loads((ROOT/'status.json').read_text())['stage'] in ('trained','evaluate','complete')
    data=np.load(ROOT/'prepared/data.npz');stats=json.loads((ROOT/'data-statistics.json').read_text())
    mapping=json.loads((ROOT/'prepared/mapping.json').read_text())
    origin_n=np.array([int((data['eyear']<=y).sum()) for y in data['test_year']])
    results={'dataset':'ICEWS05-15','seed':17,'rank_ties':'average, true gold excluded from competitors',
             'semantic_TVR':None,'UFR_status':'Fixed historical benchmark rank-worsening proxy; not adjudicated semantic error rate.',
             'arms':{}}
    costs={};shared={}
    (ROOT/'evaluation').mkdir(exist_ok=True)
    rows=[]
    for arm in ARMS:
        architecture=arm.split('_')[0]
        previous={kind:np.full(len(data['test']),np.nan) for kind in ('raw','filtered')}
        origin={kind:np.full(len(data['test']),np.nan) for kind in ('raw','filtered')}
        annual=[]
        for year in range(2005,2016):
            ids=np.flatnonzero(data['eligible']&(data['test_year']<=year))
            n=int((data['eyear']<=year).sum());oldn=int((data['eyear']<year).sum())
            model,metadata=load_model(architecture,arm,year,stats)
            cachepath=ROOT/f'evaluation/{arm}-{year}.npz'
            if year==2005 and architecture in shared:
                out=shared[architecture];costs[f'{arm}-{year}']={'seconds':0,'shared_evaluation_from':architecture+'_shared-2005'}
            elif cachepath.exists():
                out=dict(np.load(cachepath));costs[f'{arm}-{year}']={'seconds':float(out['seconds']),'reused_completed_evaluation':True}
            else:
                out=ranks(model,data,ids,n,oldn,origin_n,arm,year)
                costs[f'{arm}-{year}']={'seconds':float(out['seconds'])}
                if year==2005:shared[architecture]=out
            np.savez_compressed(cachepath,**out)
            assert np.array_equal(ids,out['ids'])
            current=data['test_year'][ids]==year;historic=~current
            newcurrent=current&((data['eyear'][data['test'][ids,0]]==year)|(data['eyear'][data['test'][ids,2]]==year))
            report={'year':year,'active_candidates':n,'new_candidates':n-oldn,'filtered':{},'raw':{},'checkpoint_metadata':metadata}
            for kind in ('raw','filtered'):
                rr=out[kind+'_full'];fixed=out[kind+'_previous_candidates'];origfixed=out[kind+'_origin_candidates']
                origin[kind][ids[current]]=rr[current]
                kr={'current':metrics(rr[current]),'current_new_entity':metrics(rr[newcurrent]),'historical':metrics(rr[historic]),
                    'step_drift':drift(previous[kind][ids[historic]],rr[historic],fixed[historic]),
                    'cumulative_origin_drift':drift(origin[kind][ids[historic]],rr[historic],origfixed[historic])}
                if kind=='filtered':
                    anchor=data['test_year'][ids]==2005
                    kr['anchor2005']=metrics(rr[anchor])
                    kr['anchor2005_drift']=drift(origin[kind][ids[anchor]],rr[anchor],origfixed[anchor])
                    # Frozen old parameters imply zero fixed-candidate rank change on all historical queries.
                    if arm.endswith('frozen_history') and historic.any():
                        invariant=bool(np.array_equal(previous[kind][ids[historic]],fixed[historic]))
                        kr['frozen_previous_candidate_rank_invariant']=invariant
                        kr['frozen_previous_candidate_max_rank_change']=float(np.max(np.abs(previous[kind][ids[historic]]-fixed[historic])))
                        assert invariant, 'frozen-history rank changed in old candidate set; inspect numerical scoring before conclusion'
                    rows.append({'arm':arm,'year':year,'current_mrr':kr['current']['mrr'],
                                 'new_entity_current_mrr':kr['current_new_entity']['mrr'],'historical_mrr':kr['historical']['mrr'],
                                 'historical_hits10':kr['historical']['hits10'],'anchor2005_mrr':kr['anchor2005']['mrr'],
                                 'anchor2005_hits10':kr['anchor2005']['hits10'],
                                 'cumulative_worsening_rate':kr['cumulative_origin_drift'].get('worsened_rate'),
                                 'step_worsening_rate':kr['step_drift'].get('worsened_rate'),
                                 'step_entity_interference_rate':kr['step_drift'].get('entity_interference_rate'),
                                 'step_mean_parameter_hrd':kr['step_drift'].get('mean_parameter_rank_contribution'),
                                 'step_mean_new_candidate_hrd':kr['step_drift'].get('mean_new_candidate_rank_contribution')})
                report[kind]=kr
                previous[kind][ids]=rr
            annual.append(report)
            results['arms'][arm]={'annual':annual}
            write_json(ROOT/'evaluation/metrics.json',results)
            write_json(ROOT/'evaluation-costs.json',costs)
            print('YEAR_EVALUATED',arm,year,json.dumps(report['filtered']),flush=True)
            if year==2015:
                anchor=data['test_year'][ids]==2005
                cases=examples(model,data,mapping,ids[anchor],origin['filtered'][ids[anchor]],out['filtered_full'][anchor],
                               out['filtered_origin_candidates'][anchor])
                write_json(ROOT/f'evaluation/{arm}-examples.json',cases)
    with (ROOT/'evaluation/annual-metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    costs['evaluation_wall_seconds']=time.perf_counter()-start
    write_json(ROOT/'evaluation-costs.json',costs)
    write_json(ROOT/'status.json',{'stage':'complete','pid':os.getpid(),'evaluation_seconds':costs['evaluation_wall_seconds']})
    print('EVALUATION_COMPLETE',costs['evaluation_wall_seconds'],flush=True)


if __name__=='__main__':
    main()
