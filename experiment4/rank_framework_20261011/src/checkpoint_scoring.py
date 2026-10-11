"""Load audited direct embeddings or saved merged incremental LoRA tables."""
import hashlib,json
from pathlib import Path
import torch
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def hash_values(x):
    if isinstance(x,dict):
        for v in x.values():yield from hash_values(v)
    elif isinstance(x,list):
        for v in x:yield from hash_values(v)
    elif isinstance(x,str) and len(x)==64:yield x

def load_tables(base,spec):
    run=base/spec['run'];audit=json.loads((base/spec['audit']).read_text());allowed=set(hash_values(audit))
    assert spec.get('exploratory') or ('PASS' in audit['status'] and not audit.get('failures'))
    bound={spec['audit']:sha(base/spec['audit'])};tables=[]
    def bind(p):
        digest=sha(p);assert digest in allowed,str(p);bound[str(p.relative_to(base))]=digest;return p
    if spec['adapter']=='direct':
        for i in range(5):
            choices=list((run/'checkpoint').glob(f'*/{i}model_best.tar'));assert len(choices)==1
            state=torch.load(bind(choices[0]),map_location='cpu',weights_only=False)['state_dict']
            tables.append((state['ent_embeddings.weight'].clone(),state['rel_embeddings.weight'].clone()))
    else:
        snaps=audit['data']['snapshots'];layers=2 if spec['method']=='FastKGE' else 20
        for i in range(5):
            ck=run/'native/seed3407/checkpoints/FB_CKGE'/f'{i}model_best.tar'
            state=torch.load(bind(ck),map_location='cpu',weights_only=True)
            if i==0:
                state=state['state_dict'];ent=state['ent_embeddings.weight'].clone();rel=state['rel_embeddings.weight'].clone()
            else:
                before=snaps[i-1]['num_ent'];now=snaps[i]['num_ent']
                feature=base/'repos'/spec['method']/'data/FB_CKGE'/str(i)/('train_distance_nodes.txt' if layers==2 else 'train_distance_nodes2.txt')
                bind(feature);values={j:(0,0.) if layers==2 else 0. for j in range(before,now)}
                for line in feature.read_text().splitlines():
                    a=line.split();k=int(a[0])
                    if k in values:values[k]=(int(a[1]),float(a[2])) if layers==2 else float(a[1])
                order=sorted(values,key=lambda k:values[k],reverse=layers==20)
                ix=torch.tensor(list(range(before))+order)
                ent=torch.cat([tables[-1][0],*[state[f'lora_ent_embeddings_list.{j}.weight'] for j in range(layers)]])[ix].clone()
                rel=torch.cat([tables[-1][1],state['lora_rel_embeddings.weight']]).clone()
            assert tuple(ent.shape)==(snaps[i]['num_ent'],200) and tuple(rel.shape)==(snaps[i]['num_rel'],200)
            tables.append((ent,rel))
    for E,R in tables:assert E.dtype==R.dtype==torch.float64 and torch.isfinite(E).all() and torch.isfinite(R).all()
    exports=[]
    for j in range(5):
        suffix='after' if spec['adapter']=='merged_lora' else 'post_train'
        sub='native/seed3407/observations' if spec['adapter']=='merged_lora' else 'ranks'
        files=[run/sub/f'train{i}_test{j}_{suffix}.npz' for i in [j,4]]
        arrays=[]
        for p in files:
            bind(p)
            with np.load(p,allow_pickle=False) as z:arrays.append(np.column_stack([z['head'],z['relation'],z['tail']]))
        assert np.array_equal(*arrays);exports.append(arrays[0])
    return tables,exports,bound

def score_queries(E,R,queries,limit):
    ids=torch.as_tensor(queries,device=E.device);q=E[ids[:,0]]+R[ids[:,1]]
    score=torch.sigmoid(9-torch.norm(q[:,None,:]-E[None,:limit,:],p=1,dim=2))
    t=ids[:,2];gold=score[torch.arange(len(ids),device=E.device),t]
    rank=1+(score>gold[:,None]).sum(1)
    masked=score.clone();masked[torch.arange(len(ids),device=E.device),t]=-torch.inf
    boundary=masked.topk(10,dim=1).values[:,-1]
    return dict(rank=rank,gold=gold,boundary=boundary,std=score.std(dim=1,unbiased=False),ties=(score==gold[:,None]).sum(1)-1)
