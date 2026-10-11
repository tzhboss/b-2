"""Independent fixed VALID utility and explicit exempted TRAIN contract costs."""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from audit_shared_graph_utility import audit as audit_valid


def query_costs(E, R, keys, golds, k):
    rows = []
    for (h,r), members in zip(keys, golds):
        scores = (E[int(h)] * R[int(r)]) @ E.T
        rank = 1 + (scores[:,None] > scores[members][None,:]).sum(0)
        # A tie may assign rank<=k to more than k candidates. Do not credit
        # that capacity trick as a contract-preserving utility improvement.
        cutoff = np.sort(scores)[-k]
        excess_tie = int((scores >= cutoff).sum()) > k
        rows.append(dict(ranks=rank, excess_boundary_tie=excess_tie))
    ranks = np.concatenate([r['ranks'] for r in rows])
    return dict(facts=len(ranks), top10_lost=int((ranks>k).sum()),
                top10_loss=float((ranks>k).mean()), raw_mrr=float((1/ranks).mean()),
                excess_boundary_tie_queries=sum(r['excess_boundary_tie'] for r in rows))


def audit(root, cfg, job):
    from common_training import sha
    from common_incremental import Data
    audit_valid(root, cfg, job)
    out = root/cfg['reaudit_directory']/job['id']; started = time.monotonic()
    try:
        can = yaml.safe_load((root/'experiments'/cfg['source_experiment']/'experiment.yaml').read_text())
        source = json.loads((root/can['config']).read_text()); assert sha(root/can['config']) == cfg['source_config_sha256']
        path = root/'artifacts/incremental'/job['id']; model = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
        init = torch.load(path/'initial.pt', map_location='cpu', weights_only=True)
        proposer = root/'artifacts/incremental'/source['source_template'].format(seed=job['seed'])
        source_model = torch.load(proposer/'best.pt', map_location='cpu', weights_only=True)
        E=model['E'].numpy(); R=model['R'].numpy(); Es=source_model['E'].double().numpy()
        assert np.array_equal(E[:7505], Es[:7505]) and np.array_equal(R, source_model['R'].double().numpy())
        scope=root/source['scope_directory']/f"shared-rank-scope-s{job['seed']}"
        policy=json.loads((scope/'receipt.json').read_text()); policy_audit=json.loads((scope/'audit.json').read_text())
        assert not policy_audit['failures'] and policy_audit['receipt_sha256']==sha(scope/'receipt.json')
        assert policy['output_sha256']['scope.npz']==sha(scope/'scope.npz')
        with np.load(scope/'scope.npz') as z:
            keys=z['exempted_query_keys'].copy(); ids=z['exempted_source_query_ids'].copy()
            members=z['pred_old_members'][ids].copy()
        diagnostic=root/source['capacity_diagnostic_directory']
        certificate=json.loads((diagnostic/'summary.json').read_text())
        assert sha(diagnostic/'certificates.npz')==certificate['output_sha256']
        with np.load(diagnostic/'certificates.npz') as z:assert np.array_equal(keys,z[f"s{job['seed']}_query_keys"])
        data=Data(root, source); truths=defaultdict(set)
        for h,r,t in data.triples(0,'train').numpy():truths[(int(h),int(r))].add(int(t))
        assert all(all(int(g) in truths[tuple(map(int,q))] for g in row) for q,row in zip(keys,members))
        all_golds=[np.asarray(sorted(truths[tuple(map(int,q))]),dtype=np.int64) for q in keys]
        controls={}
        for name,em in [('initial_old_pool',init['oldE'].numpy()),('source',Es),('after',E)]:
            controls[name]=dict(original_top10_train_truths=query_costs(em,R,keys,members,source['k']),
                                all_train_truths_at_exempt_keys=query_costs(em,R,keys,all_golds,source['k']))
        assert controls['initial_old_pool']['original_top10_train_truths']['top10_lost']==0
        reference=controls['source']['original_top10_train_truths']; actual=controls['after']['original_top10_train_truths']
        # Same SGD baseline and actual TRAIN facts, not an artificially
        # perfect preservation control with no new learning.
        train_gate=bool(actual['top10_lost']<=reference['top10_lost'] and actual['raw_mrr']>=reference['raw_mrr'])
        tie_gate=bool(actual['excess_boundary_tie_queries']==0)
        evidence={str(p):sha(p) for p in [scope/'scope.npz',scope/'receipt.json',scope/'audit.json',
                  diagnostic/'summary.json',diagnostic/'certificates.npz',proposer/'best.pt',path/'best.pt']}
        evidence.update(data.bound)
        result=dict(status='PASS_EXPLICIT_EXEMPTED_TRAIN_COST_REPLAY',failures=[],test_read=False,
                    valid_read_by_separate_common_audit=True,exempted_queries=len(keys),controls=controls,
                    nonworse_than_source_exempt_train_gate=train_gate,no_excess_boundary_ties_gate=tie_gate,
                    evidence_sha256=evidence,elapsed_s=time.monotonic()-started,
                    limits=['Endpoint cost of fixed exemptions; not causal attribution of whole optimization trajectory.',
                            'Removal changes the old TRAIN truth guarantee even when nonworse than source SGD.',
                            'All TRAIN truth metrics disclose ranks outside top10; extra cost gate uses original top10 TRAIN truths.',
                            'Exact endpoint tie check does not certify arbitrary floating-point kernels.'])
        for p,d in evidence.items():assert sha(Path(p))==d
        (out/'contract-audit.json').write_text(json.dumps(result,indent=2))
        common=json.loads((out/'audit.json').read_text()); common['capacity_contract_audit']=result
        (out/'audit.json').write_text(json.dumps(common,indent=2));print('CAPACITY_UTILITY_AUDIT_PASS',job['id'],train_gate,tie_gate,flush=True)
    except Exception as exc:
        (out/'contract-audit.json').write_text(json.dumps(dict(status='FAIL_RETAINED',failures=[repr(exc)]),indent=2));raise


def aggregate(root,cfg):
    from common_training import sha
    can=yaml.safe_load((root/'experiments'/cfg['id']/'experiment.yaml').read_text());assert 'terminal' in can['status'] and not can['active_processes']
    source_can=yaml.safe_load((root/'experiments'/cfg['source_experiment']/'experiment.yaml').read_text())
    source=json.loads((root/source_can['config']).read_text());assert sha(root/source_can['config'])==cfg['source_config_sha256']
    original=json.loads((root/cfg['original_utility_result']).read_text())
    assert not original['unavailable'];original_rows={r['method']:r for r in original['rows']}
    groups=defaultdict(list); unavailable=[]
    for job in cfg['jobs']:
        if can['jobs'][job['id']]['status']!='completed_audited':unavailable.append(dict(id=job['id'],status=can['jobs'][job['id']]['status']));continue
        row=json.loads((root/cfg['reaudit_directory']/job['id']/'audit.json').read_text())
        assert row['status']=='PASS_FULL_COMMON_VALID_METRICS' and not row['failures'] and not row['capacity_contract_audit']['failures']
        groups[job['method']].append(row)
    records=[]
    for method in dict.fromkeys(j['method'] for j in cfg['jobs']):
        rows=sorted(groups[method],key=lambda r:r['seed']);seeds=[r['seed'] for r in rows]
        if seeds!=cfg['seeds']:records.append(dict(method=method,seeds=seeds,decision='INCOMPLETE_UTILITY_EVIDENCE'));continue
        values={}
        for name,path in [('new_mrr',('after',1)),('source_new_mrr',('source',1))]:values[name]=[r['validation'][path[0]][path[1]]['raw_mrr'] for r in rows]
        values['old_loss']=[r['validation']['after_old_top10_loss'] for r in rows]
        values['source_old_loss']=[r['validation']['source_old_top10_loss'] for r in rows]
        cohort='query_head_old_gold_new'
        for metric in ['raw_mrr','hits10']:
            for arm in ['source','after']:values[arm+'_answer_'+metric]=[r['validation']['directional'][cohort][arm][metric] for r in rows]
        new_gate=np.mean(values['new_mrr'])>=source['utility_new_mrr_ratio']*np.mean(values['source_new_mrr'])
        old_gate=np.mean(values['source_old_loss'])>0 and np.mean(values['old_loss'])<=source['utility_old_loss_ratio']*np.mean(values['source_old_loss']) and all(a<=b for a,b in zip(values['old_loss'],values['source_old_loss']))
        direction_gate=np.mean(values['after_answer_raw_mrr'])>=source['utility_direction_mrr_ratio']*np.mean(values['source_answer_raw_mrr']) and np.mean(values['after_answer_hits10'])>=source['utility_direction_hits10_ratio']*np.mean(values['source_answer_hits10'])
        train_gate=all(r['capacity_contract_audit']['nonworse_than_source_exempt_train_gate'] for r in rows)
        tie_gate=all(r['capacity_contract_audit']['no_excess_boundary_ties_gate'] for r in rows)
        records.append(dict(method=method,seeds=seeds,decision='PASS_DIRECTIONAL_AND_EXPLICIT_TRAIN_COST_SCREEN' if all([new_gate,old_gate,direction_gate,train_gate,tie_gate]) else 'NO_GO_FOR_THIS_FIXED_COMBINATION',
                            new_gate=bool(new_gate),old_gate=bool(old_gate),new_answer_gate=bool(direction_gate),exempt_train_gate=train_gate,tie_gate=tie_gate,values=values,
                            contract_costs=[r['capacity_contract_audit'] for r in rows],original_control=original_rows[cfg['method_controls'][method]]))
    dest=root/'results'/cfg['id'];dest.mkdir(parents=True,exist_ok=False)
    result=dict(scope='Fixed exposed development utility with additional explicit TRAIN costs; no heldout/SOTA or novelty claim',
                original_gates_unchanged=True,additional_gates=['Each seed exempted original-top10 TRAIN loss and MRR nonworse than source SGD','No excess endpoint boundary ties at exempt keys'],
                rows=records,unavailable=unavailable,original_control_sha256=sha(root/cfg['original_utility_result']))
    (dest/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--job');p.add_argument('--stage',choices=['audit','aggregate'],default='audit')
    a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'src'));torch.set_num_threads(2);cfg=json.loads((root/a.config).read_text())
    if a.stage=='aggregate':aggregate(root,cfg)
    else:audit(root,cfg,next(j for j in cfg['jobs'] if j['id']==a.job))
