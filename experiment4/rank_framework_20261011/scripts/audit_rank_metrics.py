"""Independent scalar retention derivation from fully rescored Rank/QP observations."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_scoring import sha
from rank_metrics import transitions
def audit(root,base,spec):
    out=root/'artifacts'/spec['id'];t=json.loads((out/'terminal.json').read_text())
    for p,d in t['bound_input_sha256'].items():assert sha(p)==d
    checked=0
    for summary in t['summaries']:
        name=summary['method'];path=out/(name+'.npz');assert sha(path)==t['output_sha256'][path.name]
        with np.load(path,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        assert np.all(v['before_rank']<=10)
        lost=sum(int(a>10) for a in v['after_rank'].tolist());assert lost==summary['lost_top10']
        metrics=transitions(v['before_rank'],v['after_rank'],v['before_gold'],v['after_gold'],v['before_std']);assert all(summary[k]==x for k,x in metrics.items())
        source=base.parent/'rank_comparison_20261010/artifacts'/spec['kind'];records=json.loads((source/'per_case.json').read_text());rows=[x for x in records if x['method']==name]
        assert len(rows)==400 and np.mean([x['safe_success'] for x in rows])==summary['safe_success'] and np.mean([x['success_at_k'] for x in rows])==summary['new_hits10']
        assert sum(x['damage_count'] for x in rows)==lost
        checked+=len(v['before_rank'])
    assert t['sampled_full_candidate_scalar_rank_checks']>0
    r=dict(status='PASS_COMMON_METRICS_INHERITED_VECTOR_AUDITS',full_observation_derivation_checks=checked,
        sampled_full_candidate_scalar_rank_checks=t['sampled_full_candidate_scalar_rank_checks'],failures=[],terminal_sha256=sha(out/'terminal.json'),source_sha256=sha(__file__),limitations=t['limitations'])
    (out/'audit.json').write_text(json.dumps(r,indent=2));print('RANK_AUDIT_PASS',spec['id'],checked,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();spec=next(x for x in json.loads((a.root/'config.json').read_text())['jobs'] if x['id']==a.job);audit(a.root,a.base,spec)
