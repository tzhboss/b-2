"""Independent NumPy full-candidate old validation fixture audit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common_training import sha
def audit(root,jid):
    out=root/'artifacts/base'/jid;t=json.loads((out/'terminal-receipt.json').read_text())
    for n,d in t['output_sha256'].items():assert sha(out/n)==d
    for n,d in t['bound_input_sha256'].items():assert sha(n)==d
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);E=ck['state_dict']['entity.weight'].numpy();R=ck['state_dict']['relation.weight'].numpy()
    assert len(E)==7505 and len(R)==474 and not t['test_triples_read'] and not t['new_snapshot_triples_read']
    history=json.loads((out/'training-observations.json').read_text());choices=[r for r in history if 'validation' in r]
    best=max(choices,key=lambda r:r['validation']['raw_strict_mrr']);assert ck['epoch']==t['best_epoch']==best['epoch']
    assert len(history)==t['epochs_executed'] and all(np.isfinite(r['train_loss']) for r in history)
    if ck['kind']=='TransE':E=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-12);R=R/np.maximum(np.linalg.norm(R,axis=1,keepdims=True),1e-12)
    with np.load(out/'valid-fixture.npz',allow_pickle=False) as z:q=z['queries'];ref=z['scores'];ranks=z['ranks']
    maxerr=0.;rank_mismatches=0;tol=1e-12 if ck['kind']=='TransE' else 2e-4
    for i,(h,r,tail) in enumerate(q):
        if ck['kind']=='TransE':scores=1/(1+np.exp(-9+np.abs(E[h]+R[r]-E).sum(1)))
        else:scores=(E[h].astype(np.float64)*R[r].astype(np.float64))@E.astype(np.float64).T
        err=float(np.max(np.abs(scores-ref[i])));maxerr=max(maxerr,err);assert err<=tol,(jid,i,err)
        rank=1+sum(v>scores[tail] for v in scores.tolist());rank_mismatches+=int(rank!=ranks[i])
    # Exact rank gate; score tolerance cannot excuse rank disagreement.
    assert rank_mismatches==0,(jid,rank_mismatches)
    report=dict(status='PASS_COMMON_BASE_VALID_FIXTURE',failures=[],queries=len(q),full_candidate_score_checks=len(q)*len(E),max_abs_score_error=maxerr,rank_mismatches=rank_mismatches,
        terminal_sha256=sha(out/'terminal-receipt.json'),auditor_sha256=sha(Path(__file__)),limits=['64 old VALID full-candidate rows independently rescored; not independent training rerun.'])
    (out/'audit.json').write_text(json.dumps(report,indent=2));print('BASE_AUDIT_PASS',jid,maxerr,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args();audit(a.root.resolve(),a.job)
