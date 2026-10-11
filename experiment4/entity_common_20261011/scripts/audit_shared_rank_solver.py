"""Independent scalar saved-model and analytic toy solution audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import yaml


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root,cfg):
    can=root/'experiments'/cfg['id']/'experiment.yaml';s=yaml.safe_load(can.read_text());assert s['status']=='completed_pending_independent_check'
    for name,digest in s['frozen_input_sha256'].items():assert sha(root/name)==digest
    out=root/cfg['output_directory'];r=json.loads((out/'receipt.json').read_text());assert r['passed'] and not r['test_read'] and not r['graph_facts_read']
    with np.load(out/'synthetic_models.npz') as z:E=z['initial'];R=z['relations'];joint=z['joint'];keys=z['keys'];refs=z['refs']
    assert E.shape==joint.shape==(13,2) and R.shape==(2,2)
    ranks=[]
    def rank(h,rel,g):
        scores=[sum(float(joint[h,d])*float(R[rel,d])*float(joint[c,d]) for d in range(2)) for c in range(13)]
        return 1+sum(v>scores[g] for v in scores)
    for q,g in refs:ranks.append(rank(*keys[q],g))
    assert max(ranks)<=10 and rank(0,0,12)==r['new_gold_rank']==8
    untouched=[i for i in range(13) if i not in [8,12]];assert np.array_equal(E[untouched],joint[untouched])
    margin=cfg['solver']['guard_margin'];analytic=np.array([[(3+np.sqrt(9-48*margin))/2,9.],[5.,-1.]])
    error=float(np.max(np.abs(joint[[8,12]]-analytic)));assert error<cfg['analytic_tolerance']
    assert r['jacobian_cases']==len(r['cases'])==cfg['jacobian_cases']
    assert max(c['error'] for c in r['cases'])==r['max_derivative_error']<cfg['derivative_tolerance']
    assert r['joint']['accepted'] and r['joint']['cuts']==3 and r['joint']['cut_rounds']==1
    if cfg['id'].endswith('R2'):assert r['frozen_control']['accepted'] and 0<r['frozen_control']['movements'][0]<6
    else:assert not r['frozen_control']['accepted']
    audit=dict(status='PASS_INDEPENDENT_SCALAR_SHARED_RANKS_AND_ANALYTIC_TOY',failures=[],all_reference_ranks=ranks,
               analytic_toy_vector_error=error,new_gold_rank=8,unmodified_rows_exact=True,
               saved_model_sha256=sha(out/'synthetic_models.npz'),receipt_sha256=sha(out/'receipt.json'),auditor_sha256=sha(Path(__file__)),
               limits=['Synthetic feasibility only; no real graph optimizer utility or global optimum claim.',
                       'Independent derivatives exercised by control producer, saved-model scores by separate scalar audit.'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2));formal=root/'results'/cfg['id'];formal.mkdir(parents=True,exist_ok=False)
    (formal/'summary.json').write_text(json.dumps(dict(result=r,audit=audit),indent=2))
    s.update(status='completed_audited_synthetic_solver_controls_only',result_directory=str(formal.relative_to(root)));can.write_text(yaml.safe_dump(s,sort_keys=False));print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();run(a.root.resolve(),json.loads((a.root/a.config).read_text()))
