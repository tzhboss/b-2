"""Full saved observation derivation and native replay receipt review."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_scoring import sha
from rank_metrics import transitions
def audit(root,base,spec):
    out=root/'artifacts'/spec['id'];term=json.loads((out/'terminal.json').read_text())
    for name,d in term['output_sha256'].items():assert sha(out/name)==d
    for p,d in term['bound_input_sha256'].items():assert sha(p)==d
    with np.load(out/'rank-transitions.npz',allow_pickle=False) as z:v={k:z[k] for k in z.files}
    n=term['batches'];assert len(v['role'])==10*n and term['native_before_after_record_checks']==20*n
    old=v['role']=='locality';edit=v['role']=='edit';assert old.sum()==8*n and edit.sum()==2*n
    counts=dict(queries=0,old_top10=0,lost_top10=0,worsened=0,unchanged=0,improved=0);safes=[]
    for b,a in zip(v['before_rank'][old],v['after_rank'][old]):
        counts['queries']+=1;counts['old_top10']+=int(b<=10);counts['lost_top10']+=int(b<=10<a)
        counts['worsened']+=int(a>b);counts['unchanged']+=int(a==b);counts['improved']+=int(a<b)
    for i in range(n):
        ix=v['batch_idx']==i;assert ix.sum()==10
        b=v['before_rank'][ix];a=v['after_rank'][ix];safes.append(bool(all(a[8:]<=10) and all(a[:8][b[:8]<=10]<=10)))
    summary=term['summary'];assert all(summary[k]==v for k,v in counts.items())
    assert np.mean(safes)==summary['safe_success'] and np.mean(v['after_rank'][edit]<=10)==summary['new_hits10']
    metrics=transitions(v['before_rank'][old],v['after_rank'][old],v['before_gold'][old],v['after_gold'][old],v['before_std'][old]);assert all(summary[k]==x for k,x in metrics.items())
    assert term['max_native_export_logit_error']<=1e-4 and term['parameters_unchanged'] and term['inputs_unchanged']
    result=dict(status='PASS_FULL_NATIVE_REPLAY_WITH_RESTORE_LIMITATION',observations=10*n,full_native_before_after_checks=20*n,
        strict_rank_cpu_gpu_crosschecks=20*n,failures=[],terminal_sha256=sha(out/'terminal.json'),source_sha256=sha(__file__),
        limitations=term['limitations']+['Audit independently derives aggregates from saved ranks; full strict scores CPU/GPU crosschecked during replay, original native sort/logits verified on every batch.'])
    (out/'audit.json').write_text(json.dumps(result,indent=2));print('LM_AUDIT_PASS',spec['id'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();spec=next(x for x in json.loads((a.root/'config.json').read_text())['jobs'] if x['id']==a.job);audit(a.root,a.base,spec)
