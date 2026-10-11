"""Diagnose native MLM rank differences without changing source audit decisions."""
import json,time
from pathlib import Path
import numpy as np
import torch
from common_training import sha
from lm_common_base import cached_npz,base_validate
from lm_editor_common_v2 import load_model,setup_seed


def inspect(root,cfg):
    started=time.monotonic();setup_seed(cfg['seed'])
    base=root/cfg['base_path'];fixture_path=base.with_name('valid-fixture.npz')
    original_path=base.with_name('audit.json');original=json.loads(original_path.read_text())
    hashes={str(p.relative_to(root)):sha(p) for p in (base,fixture_path,original_path)}
    f=cached_npz(fixture_path);expected=f['scores'];gold_ids=f['queries'][:,2]
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False)
    model=load_model(root,cfg,base,'cpu');replays={};disputed=set()
    for device,batch,dtype in cfg['replays']:
        model=model.to(device=device,dtype=getattr(torch,dtype)).eval()
        _,scores,ranks=base_validate(model,f,torch.device(device),batch_size=batch)
        expected_gold=expected[np.arange(len(expected)),gold_ids];gold=scores[np.arange(len(scores)),gold_ids]
        ids=np.where(ranks!=f['ranks'])[0];disputed.update(ids.tolist())
        details=[]
        for row in ids:
            flips=np.where((scores[row]>gold[row])!=(expected[row]>expected_gold[row]))[0]
            details.append(dict(fixture_row=int(row),query=f['queries'][row].tolist(),original_rank=int(f['ranks'][row]),replay_rank=int(ranks[row]),
                                flipped_candidates=[dict(entity=int(c),original_margin=float(expected[row,c]-expected_gold[row]),replay_margin=float(scores[row,c]-gold[row])) for c in flips]))
        label=f'{device}-{dtype}-b{batch}'
        replays[label]=dict(max_abs_logit_error=float(np.abs(scores-expected).max()),rank_mismatches=len(ids),details=details)
        np.savez_compressed(out/(label.replace(':','_')+'.npz'),scores=scores,ranks=ranks)
        print('NUMERICAL_REPLAY',label,json.dumps(replays[label]),flush=True)
    # Source decisions and checkpoints are immutable; this is a diagnostic.
    assert all(sha(root/p)==h for p,h in hashes.items())
    result=dict(status='COMPLETED_NUMERICAL_DIAGNOSTIC_ONLY',source_audit_status=original['status'],source_sha256=hashes,
                original_audit_unchanged=True,replays=replays,disputed_fixture_rows=sorted(disputed),elapsed_s=time.monotonic()-started,
                limits=['No audit gate relaxed or source FAIL overwritten.','Float64 native inference is numerical evidence, not independent architecture reproduction.','No new TRAIN, TEST, meta training or online editing in this diagnostic.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2));return result
