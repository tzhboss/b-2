"""Config-driven unfiltered Rank metrics on existing continual checkpoints."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_scoring import load_tables,score_queries,sha
from rank_metrics import transitions

def evaluate(base,root,spec):
    tables,queries,bound=load_tables(base,spec);out=root/'artifacts'/spec['id'];out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device=torch.device('cuda:0');start=time.monotonic()
    final=[torch.nn.functional.normalize(x.to(device),p=2,dim=-1) for x in tables[4]]
    records=[];pair_lists={k:[] for k in ['before_rank','after_rank','expanded_rank','before_gold','after_gold','before_std']}
    for cohort in range(5):
        q=queries[cohort];before=[torch.nn.functional.normalize(x.to(device),p=2,dim=-1) for x in tables[cohort]]
        values={};limit=len(before[0])
        with torch.no_grad():
            for off in range(0,len(q),32):
                assert time.monotonic()-start<7200,'Fixed per-job budget'
                block=q[off:off+32]
                b=score_queries(*before,block,limit);a=score_queries(*final,block,limit);e=score_queries(*final,block,len(final[0]))
                for prefix,data in [('before',b),('after',a),('expanded',e)]:
                    for key,v in data.items():values.setdefault(prefix+'_'+key,[]).append(v.cpu().numpy())
                assert torch.all(e['rank']>=a['rank'])
                if off%12800==0:print('SCORE_PROGRESS',spec['id'],cohort,off,len(q),flush=True)
        values={k:np.concatenate(v) for k,v in values.items()};values['queries']=q
        np.savez_compressed(out/f'cohort{cohort}.npz',**values)
        if cohort<4:
            records.append(dict(cohort=cohort,old_candidates=limit,final_candidates=len(final[0]),
              fixed=transitions(values['before_rank'],values['after_rank'],values['before_gold'],values['after_gold'],values['before_std']),
              expansion_only=transitions(values['after_rank'],values['expanded_rank'],values['after_gold'],values['expanded_gold'],values['after_std'])))
            for key in pair_lists:pair_lists[key].append(values[key])
        else:new_hits=float(np.mean(values['expanded_rank']<=10))
        del before;torch.cuda.empty_cache()
    allv={k:np.concatenate(v) for k,v in pair_lists.items()}
    common=dict(method=spec['method'],dataset=spec['dataset'],model=spec.get('model','TransE'),
      update_unit='native snapshot0..3 learning to snapshot4',probe_split='native old TEST head+tail observations',
      evidence_status='EXPLORATORY_UPSTREAM_INTEGRITY_FAIL' if spec.get('exploratory') else 'PENDING_AUDIT',
      new_hits10=new_hits,safe_success=None,scope='all exported old queries; fixed historical entity pools; exposed data')
    summary=dict(**common,candidate_contract='same_old_candidates',**transitions(allv['before_rank'],allv['after_rank'],allv['before_gold'],allv['after_gold'],allv['before_std']))
    expansion=dict(**common,candidate_contract='expansion_only_same_final_weights',**transitions(allv['after_rank'],allv['expanded_rank'],allv['after_gold'],allv['after_gold'],np.ones(len(allv['after_rank']))))
    for n,d in bound.items():assert sha(base/n)==d,n
    result=dict(id=spec['id'],status='TERMINAL_PENDING_AUDIT',summary=summary,expansion=expansion,cohorts=records,bound_input_sha256=bound,
      output_sha256={p.name:sha(p) for p in out.glob('*.npz')},implementation_sha256=sha(__file__),
      elapsed_s=time.monotonic()-start,runtime=dict(torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0)),
      score_contract='native normalized float64 TransE sigmoid9-L1; unfiltered strict greater; no new weights/training')
    (out/'terminal.json').write_text(json.dumps(result,indent=2));print('TERMINAL',spec['id'],summary['old_top10_loss_fraction'],new_hits,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();spec=next(x for x in json.loads((a.root/'config.json').read_text())['jobs'] if x['id']==a.job);evaluate(a.base.resolve(),a.root.resolve(),spec)
