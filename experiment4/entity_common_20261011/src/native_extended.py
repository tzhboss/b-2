"""Source-faithful IncDE training and SAGE learned dimensional growth bridges."""
import ast,importlib,json,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from common_training import sha
from common_incremental import Data,defaults,scores,ranks,metrics

def load_native(root,cfg,job):
    torch.set_num_threads(2);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    random.seed(job['seed']);np.random.seed(job['seed']);torch.backends.cuda.matmul.allow_tf32=False
    data=Data(root,cfg);train=[data.triples(s,'train') for s in (0,1)];device=torch.device('cuda:0')
    source=(data.root/cfg['native_sources'][job['method']]).resolve()
    manifest=json.loads((source.parent/'copied-source-sha256.json').read_text())
    bound={}
    for f in [*sorted((source/'src').rglob('*.py')),source/'main.py']:
        digest=sha(f);relative='worktree/'+str(f.relative_to(source));assert manifest[relative]==digest
        bound[str(f)]=digest
    for dep in cfg['dependency_paths']:sys.path.insert(0,str((data.root/dep).resolve()))
    sys.path.insert(0,str(source))
    args=NS(**defaults(source/'src/parse_args.py'))
    args.__dict__.update(snapshot=0,snapshot_test=0,snapshot_valid=0,snapshot_num=2,epoch=0,device=device,
                         seed=job['seed'],random_seed=job['seed'],gpu=0,batch_size=cfg['batch_size'],neg_ratio=cfg['negatives'],
                         margin=cfg['margin'],learning_rate=job['learning_rate'])
    args.__dict__.update(job.get('native_options',{}))
    snapshots={}
    for s,q in enumerate(train):
        original=q[::2].tolist();alltrain=(train[0][::2] if s==0 else torch.cat([train[0][::2],train[1][::2]])).tolist()
        snapshots[s]=NS(num_ent=7505 if s==0 else 11258,num_rel=474,train=original,train_new=original,
                        train_all=alltrain,edge_index=q[:,[0,2]].T.to(device),edge_type=q[:,1].to(device))
    kg=NS(snapshots=snapshots)
    module='DLKGE' if job['method']=='IncDE' else 'LKGE'
    model=importlib.import_module('src.model.'+module).TransE(args,kg)
    bpath=root/'artifacts/base'/f"{job['base']}-s{job['seed']}"/'best.pt'
    audit=json.loads(bpath.with_name('audit.json').read_text());assert not audit['failures']
    base=torch.load(bpath,map_location=device,weights_only=True)
    E0=base['state_dict']['entity.weight'];R0=base['state_dict']['relation.weight']
    with torch.no_grad():model.ent_embeddings.weight.copy_(E0);model.rel_embeddings.weight.copy_(R0)
    bound[str(bpath)]=sha(bpath)
    processor_class=importlib.import_module('src.model.model_process').TrainBatchProcessor
    return data,train,args,kg,model,processor_class,E0,R0,bound,source

def source_mapping_function(path):
    tree=ast.parse(path.read_text())
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='train_mapping_matrix')
    # Execute only the author's saved method; do not import main.py's argument
    # parsing, TEST routines, destructive path setup or five-snapshot driver.
    namespace=dict(torch=torch,np=np)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['train_mapping_matrix']

def train_native(root,cfg,job,preflight=False):
    started=time.monotonic();stage='native-preflight' if preflight else 'incremental'
    out=root/'artifacts'/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    data,train,args,kg,model,processor_class,E0,R0,bound,source=load_native(root,cfg,job)
    extra={};device=E0.device
    if job['method']=='SAGE':
        sampler=processor_class(args,kg)
        selected=sampler.evaluate_on_training_data(model)
        assert selected and all(max(int(h),int(t))<7505 for h,r,t,label in selected)
        np.savez_compressed(out/'mapping-train.npz',triples=np.array([[int(h),int(r),int(t)] for h,r,t,label in selected]))
        model.mapping=nn.Linear(180,30,bias=False,dtype=torch.float64).to(device)
        nn.init.xavier_normal_(model.mapping.weight)
        model.this_snapshot_ent_embeddings=model.ent_embeddings.weight.clone()
        model.this_snapshot_rel_embeddings=model.rel_embeddings.weight.clone()
        source_mapping_function(source/'main.py')(NS(model=model,args=args,kg=kg),selected)
        with torch.no_grad():
            Emap=torch.cat([E0,model.mapping(E0)],1);Rmap=torch.cat([R0,model.mapping(R0)],1)
            model.ent_embeddings=nn.Embedding(7505,210,dtype=torch.float64,device=device)
            model.rel_embeddings=nn.Embedding(474,210,dtype=torch.float64,device=device)
            model.ent_embeddings.weight.copy_(Emap);model.rel_embeddings.weight.copy_(Rmap)
        extra.update(mapping_samples=len(selected),mapping_epochs=args.new_dim_epoch_num,mapping_source_sha256=sha(source/'main.py'),
                     mapping_elapsed_s=time.monotonic()-started,old_dimension=180,new_dimension=210,
                     upstream_source_limitation='Prior SAGE original producer metadata integrity FAIL retained; scientific copied-source manifest verified unchanged.')
        model.switch_snapshot();args.emb_dim=210
    else:
        receipt=json.loads((root/cfg['feature_directory']/'receipt.json').read_text())
        assert receipt['status']=='PASS_PARTITIONED_EXACT_TRAIN_ONLY'
        for name,digest in receipt['output_sha256'].items():assert sha(root/cfg['feature_directory']/name)==digest
        # Author trainer addresses data_path + snapshot + '/feature'.
        args.data_path=str((root/cfg['feature_directory']).parent)+'/'
        bound[str(root/cfg['feature_directory']/'receipt.json')]=sha(root/cfg['feature_directory']/'receipt.json')
        for name,digest in receipt['output_sha256'].items():bound[str(root/cfg['feature_directory']/name)]=digest
        model.switch_snapshot()
    args.snapshot=args.snapshot_test=args.snapshot_valid=1;args.epoch=0
    dim=model.ent_embeddings.weight.shape[1]
    gen=torch.Generator(device=device).manual_seed(job['seed']+cfg['expansion_seed_offset'])
    common_initial=torch.empty((11258,dim),dtype=torch.float64,device=device);nn.init.xavier_normal_(common_initial,generator=gen)
    with torch.no_grad():model.ent_embeddings.weight[7505:].copy_(common_initial[7505:])
    initialE,initialR=model.embedding('Train');initialE=initialE.detach().clone();initialR=initialR.detach().clone()
    # Source IncDE creates mask-weight layer before resetting its optimizer.
    model.pre_snapshot();opt=torch.optim.Adam(model.parameters(),lr=job['learning_rate'],weight_decay=0.)
    processor=processor_class(args,kg)
    valid=[data.triples(s,'valid') for s in (0,1)];vg=torch.Generator().manual_seed(cfg['validation_seed'])
    valid=[q[torch.randperm(len(q),generator=vg)[:cfg['validation_queries_per_split']]] for q in valid]
    epochs=(cfg['incde_preflight_epochs'] if job['method']=='IncDE' else 1) if preflight else cfg['epochs']
    records=[];best=-1.;best_epoch=0
    for epoch in range(epochs):
        args.epoch=epoch
        assert time.monotonic()-started<cfg['budget_seconds_per_job'],'Registered total native job budget exceeded'
        loss=processor.process_epoch(model,opt);assert np.isfinite(loss)
        rec=dict(epoch=epoch+1,loss=loss,train_directed_rows=len(train[1]))
        if preflight or epoch==0 or (epoch+1)%cfg['validation_every']==0 or epoch+1==epochs:
            model.eval();E,R=model.embedding('Valid');q=valid[1][:8].to(device)
            with torch.no_grad():assert torch.allclose(model.predict(q[:,0],q[:,1],stage='Valid'),scores(E,R,q,'TransE'),atol=1e-12,rtol=1e-12)
            v=[metrics(ranks(E,R,q[:16] if preflight else q,'TransE')) for q in valid]
            value=.5*sum(x['raw_mrr'] for x in v);rec.update(validation=v,selection_score=value)
            if value>best:
                best=value;best_epoch=epoch+1
                torch.save(dict(E=E.detach().cpu(),R=R.detach().cpu(),kind='TransE',seed=job['seed'],epoch=epoch+1,
                                method=job['method'],stage=stage,native_state={k:v.detach().cpu() for k,v in model.state_dict().items()}),out/'best.pt')
            print('NATIVE_PROGRESS',job['id'],epoch+1,round(value,6),round(time.monotonic()-started,2),flush=True)
        records.append(rec);(out/'training.json').write_text(json.dumps(records,indent=2))
    bound.update(data.bound)
    for path,digest in bound.items():assert sha(path)==digest
    torch.save(dict(E=initialE.cpu(),R=initialR.cpu(),oldE=initialE[:7505].cpu(),original_oldE=E0.cpu(),original_oldR=R0.cpu()),out/'initial.pt')
    receipt=dict(id=job['id'],method=job['method'],kind='TransE',seed=job['seed'],stage=stage,test_read=False,
                 best_epoch=best_epoch,best_validation_score=best,epochs_executed=epochs,elapsed_s=time.monotonic()-started,
                 parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(out/'best.pt'),bound_input_sha256=bound,
                 native_options={k:str(v) for k,v in args.__dict__.items()},native_extras=extra)
    (out/'training-receipt.json').write_text(json.dumps(receipt,indent=2));print('NATIVE_TERMINAL',job['id'],flush=True)

def evaluate_native(root,cfg,job):
    started=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=root/'artifacts/incremental'/job['id'];data=Data(root,cfg);queries=[data.triples(s,'test') for s in (0,1)]
    c=torch.load(out/'best.pt',map_location='cuda:0',weights_only=True);initial=torch.load(out/'initial.pt',map_location='cuda:0',weights_only=True)
    E,R=c['E'],c['R'];old_before=ranks(initial['original_oldE'],initial['original_oldR'],queries[0],'TransE')
    old_expanded=ranks(initial['E'],initial['R'],queries[0],'TransE');old_after=ranks(E,R,queries[0],'TransE');new_after=ranks(E,R,queries[1],'TransE')
    np.savez_compressed(out/'ranks.npz',old_queries=queries[0].numpy(),new_queries=queries[1].numpy(),old_before=old_before,
                        old_expanded=old_expanded,old_after=old_after,new_after=new_after)
    summary=dict(id=job['id'],method=job['method'],kind='TransE',seed=job['seed'],candidates=11258,dimension=E.shape[1],
        old_before=metrics(old_before),old_after=metrics(old_after),new_after=metrics(new_after),old_top10_protected=int((old_before<=10).sum()),
        old_top10_loss=float((old_after[old_before<=10]>10).mean()),same_pool_top10_loss=float((old_after[old_expanded<=10]>10).mean()),
        mean_rank_displacement=float((old_after-old_before).mean()),old_parameter_l2=float(torch.linalg.vector_norm(E[:7505]-initial['oldE']).cpu()),
        relation_parameter_l2=float(torch.linalg.vector_norm(R-initial['R']).cpu()),evaluation_elapsed_s=time.monotonic()-started)
    q=torch.cat([queries[0][:32],queries[1][:32]])
    with torch.no_grad():s=torch.cat([scores(E,R,x.to(E.device),'TransE').cpu() for x in q.split(8)]).numpy()
    np.savez_compressed(out/'audit-fixture.npz',queries=q.numpy(),scores=s)
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    (out/'evaluation-receipt.json').write_text(json.dumps(dict(test_input_sha256=data.bound,output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()}),indent=2))
    print('NATIVE_EVALUATION_TERMINAL',job['id'],summary,flush=True)
