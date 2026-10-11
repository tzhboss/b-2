"""Full native batch replay exposing common strict-rank metrics for LM KGE editors."""
import argparse,importlib.util,json,sys,time,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rank_metrics import transitions

def evaluate(root,base,spec):
    import torch
    from transformers import BertTokenizer
    from higher.patch import monkeypatch
    torch.set_num_threads(2);start=time.monotonic()
    helper_path=base/'code/replay_kgeditor_samples_v3_native_alias.py'
    # Generalize only the old sixteen-batch lineage count assertion; all checks stay exact.
    source=helper_path.read_text();needle='require(checked==160,"sample text lineage must bind160source examples")'
    assert source.count(needle)==1
    source=source.replace(needle,'require(checked==10*len(selected),"full selected text lineage count")')
    helper=types.ModuleType('native_replay');helper.__file__=str(helper_path);exec(compile(source,str(helper_path),'exec'),helper.__dict__)
    out=root/'artifacts'/spec['id'];run=base/spec['run'];source_audit=base/spec['audit']
    args=types.SimpleNamespace(root=base,run=run,source_audit=source_audit,work_dir=out,device='cuda:0')
    _,_,job,evidence,checkpoint,copied,dev,stable,_,_,bound=helper.prepare_contract(args)
    work,snapshot,snap_hashes=helper.isolate(args,run,copied)
    count=int(evidence['recalculated_metrics']['edit']['count'])//2
    selected=helper.selected_rows(run/'native-exposed-examples.jsonl',range(count),count)
    helper.verify_sample_text_lineage(base/'data/KGEditor-HF'/job['dataset'],dev,stable,selected)
    tokenizer=BertTokenizer(vocab_file=str(run/'base_tokenizer/vocab.txt'),do_lower_case=True);tokenizer.sanitize_special_tokens()
    entities=14951 if job['dataset']=='E-FB15k237' else 40943
    device=torch.device('cuda:0');model,learner,metadata=helper.instantiate(snapshot,checkpoint,job,device)
    initial=dict(model=helper.state_hash(model),learner=helper.state_hash(learner))
    versions=dict(model=helper.versions(model),learner=helper.versions(learner))
    values={k:[] for k in ['before_rank','after_rank','before_gold','after_gold','before_std','role','batch_idx','target']}
    safe=[];maxerr=0.;checked=0;tie_rows=0
    for ix in range(count):
        assert time.monotonic()-start<7200,'Fixed per-job budget'
        rows=selected[ix];batch,_=helper.build_inputs(rows,dev,stable,tokenizer,entities,torch,device)
        inp=helper.input_hash(batch)
        before,after,mask=helper.independent_edit_forward(model,learner,batch,30522,30522+entities,monkeypatch)
        assert inp==helper.input_hash(batch) and versions['model']==helper.versions(model) and versions['learner']==helper.versions(learner)
        rr={};targets=batch['targets'];idx=torch.arange(10,device=device)
        for label,logits in [('before',before),('after',after)]:
            score=logits[mask[0],mask[1],30522:30522+entities];assert torch.isfinite(score).all()
            gold=score[idx,targets];ranks=1+(score>gold[:,None]).sum(1)
            cpu=score.detach().cpu().numpy();targets_cpu=targets.cpu().numpy()
            cpu_gold=cpu[np.arange(10),targets_cpu]
            assert np.array_equal(1+np.sum(cpu>cpu_gold[:,None],axis=1),ranks.cpu().numpy())
            order=torch.sort(score,dim=1,descending=True)[1];native=1+order.argsort(dim=1)[idx,targets]
            for pos,row in enumerate(rows):
                err=abs(float(gold[pos])-row['target_logit_'+label]);maxerr=max(maxerr,err)
                assert err<=1e-4 and int(native[pos])==row['rank_'+label] and order[pos,:3].cpu().tolist()==row['top3_'+label],(ix,pos,label,err)
            values[label+'_rank'].extend(ranks.cpu().tolist());values[label+'_gold'].extend(gold.cpu().tolist())
            if label=='before':values['before_std'].extend(score.double().std(dim=1,unbiased=False).cpu().tolist())
            tie_rows+=int(((score==gold[:,None]).sum(1)>1).sum());rr[label]=ranks.cpu().numpy();checked+=10
        old=rr['before'][:8]<=10
        safe.append(bool(np.all(rr['after'][8:]<=10) and np.all(rr['after'][:8][old]<=10)))
        values['role'].extend([x['role'] for x in rows]);values['batch_idx'].extend([ix]*10);values['target'].extend(targets.cpu().tolist())
        if (ix+1)%100==0:print('EDITOR_PROGRESS',spec['id'],ix+1,count,round(time.monotonic()-start,2),flush=True)
        del batch,before,after,mask,score,order,gold,ranks,native
    arrays={k:np.asarray(v) for k,v in values.items()};np.savez_compressed(out/'rank-transitions.npz',**arrays)
    old=arrays['role']=='locality';edit=arrays['role']=='edit'
    summary=dict(method='KGEditor',dataset=spec['dataset'],model='EXBert',update_unit='native two-edit batch with8 repeated locality probes',probe_split='native stable.jsonl exposed probes',
       candidate_contract='same_entity_vocabulary',evidence_status='PENDING_AUDIT_NATIVE_RESTORE_LIMITATION',
       new_hits10=float(np.mean(arrays['after_rank'][edit]<=10)),safe_success=float(np.mean(safe)),
       **transitions(arrays['before_rank'][old],arrays['after_rank'][old],arrays['before_gold'][old],arrays['after_gold'][old],arrays['before_std'][old]))
    assert initial==dict(model=helper.state_hash(model),learner=helper.state_hash(learner))
    assert versions==dict(model=helper.versions(model),learner=helper.versions(learner))
    for p,d in bound.items():assert helper.sha(p)==d
    for p,d in snap_hashes.items():assert helper.sha(out/p)==d
    terminal=dict(id=spec['id'],status='TERMINAL_PENDING_AUDIT',summary=summary,batches=count,observations=10*count,
      native_before_after_record_checks=checked,max_native_export_logit_error=maxerr,exact_tie_observations=tie_rows,
      parameters_unchanged=True,inputs_unchanged=True,architecture=metadata,bound_input_sha256=bound,
      helper_sha256=helper.sha(helper_path),implementation_sha256=helper.sha(__file__),elapsed_s=time.monotonic()-start,
      output_sha256={'rank-transitions.npz':helper.sha(out/'rank-transitions.npz')},
      limitations=['Native CPU shared-embedding restoration retained; serialized base embeddings can be overwritten by learner storage.',
        'edit_test used in checkpoint selection. Full repeated native batches, not independent full unique L-Test or global safety.',
        'Joint SafeSucc refers to two edits and eight probes; Rank single-request relation TRAIN scope differs.'])
    (out/'terminal.json').write_text(json.dumps(terminal,indent=2));print('EDITOR_TERMINAL',spec['id'],summary['old_top10_loss_fraction'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();spec=next(x for x in json.loads((a.root/'config.json').read_text())['jobs'] if x['id']==a.job);evaluate(a.root.resolve(),a.base.resolve(),spec)
