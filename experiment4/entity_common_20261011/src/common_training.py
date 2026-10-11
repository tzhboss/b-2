"""Common old-graph training for subsequent native-method task adapters."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def old_data(root,cfg):
    base=(root/cfg['source_root']).resolve();folder=base/cfg['dataset']/'0'
    audit_path=base/cfg['source_audit'];audit=json.loads(audit_path.read_text())
    assert not audit['failures'] and not audit['missing']
    bound={str(audit_path):sha(audit_path)}
    def read(name):
        p=folder/name;digest=sha(p);assert digest==audit['data']['sha256']['0/'+name]
        bound[str(p)]=digest;return p.read_text().splitlines()
    entities={k:int(v) for k,v in (x.split() for x in read('entity2id.txt'))}
    relations={k:int(v) for k,v in (x.split() for x in read('relation2id.txt'))}
    def triples(name):
        data=[]
        for line in read(name):
            h,r,t=line.split();h,r,t=entities[h],relations[r],entities[t]
            assert r%2==0;data.append((h,r,t));data.append((t,r+1,h))
        return torch.tensor(data,dtype=torch.long)
    train=triples('train.txt');valid=triples('valid.txt')
    assert len(entities)==7505 and len(relations)==474
    # Deliberately no TEST or snapshot1 triple reader in this training module.
    return len(entities),len(relations),train,valid,bound

class EmbeddingModel(nn.Module):
    def __init__(self,nent,nrel,dim,kind,dtype):
        super().__init__();self.kind=kind
        self.entity=nn.Embedding(nent,dim,dtype=dtype);self.relation=nn.Embedding(nrel,dim,dtype=dtype)
        if kind=='TransE':
            nn.init.xavier_normal_(self.entity.weight);nn.init.xavier_normal_(self.relation.weight)
        else:
            nn.init.normal_(self.entity.weight,std=.12);nn.init.normal_(self.relation.weight,std=.12)
    def triple(self,h,r,t):
        H=self.entity(h);R=self.relation(r);T=self.entity(t)
        if self.kind=='TransE':
            return (F.normalize(H,dim=-1)+F.normalize(R,dim=-1)-F.normalize(T,dim=-1)).abs().sum(-1)
        return (H*R*T).sum(-1)
    def candidate_scores(self,queries):
        h,r=queries[:,0],queries[:,1]
        if self.kind=='TransE':
            E=F.normalize(self.entity.weight,dim=-1);R=F.normalize(self.relation.weight,dim=-1)
            q=E[h]+R[r];return torch.sigmoid(9-(q[:,None,:]-E[None,:,:]).abs().sum(-1))
        return (self.entity(h)*self.relation(r))@self.entity.weight.T

@torch.no_grad()
def validate(model,queries,device):
    model.eval();ranks=[]
    for chunk in queries.split(32):
        x=chunk.to(device);s=model.candidate_scores(x);gold=s[torch.arange(len(x),device=device),x[:,2]]
        ranks.extend((1+(s>gold[:,None]).sum(1)).cpu().tolist())
    a=np.asarray(ranks);return dict(raw_strict_mrr=float(np.mean(1/a)),raw_hits10=float(np.mean(a<=10)),queries=len(a))

def train_old(root,cfg,job):
    seed=job['seed'];torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');assert torch.cuda.is_available()
    nent,nrel,train,valid,bound=old_data(root,cfg)
    gen=torch.Generator().manual_seed(seed);dtype=torch.float64 if job['kind']=='TransE' else torch.float32
    model=EmbeddingModel(nent,nrel,job['dim'],job['kind'],dtype).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=job['learning_rate'])
    vgen=torch.Generator().manual_seed(cfg['validation_query_selection_seed'])
    val_ids=torch.randperm(len(valid),generator=vgen)[:cfg['validation_queries']]
    selected=valid[val_ids];out=root/'artifacts/base'/job['id'];out.mkdir(parents=True,exist_ok=False)
    best=-1.;records=[];start=time.monotonic();train=train.to(device);checks=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[]
        for indices in torch.randperm(len(train),generator=gen).split(cfg['batch_size']):
            assert time.monotonic()-start<cfg['budget_seconds_per_job'],'Registered wall clock budget exceeded'
            pos=train[indices.to(device)];h,r,t=pos.unbind(1)
            neg=torch.randint(nent,(len(pos),cfg['negatives']),generator=gen).to(device)
            p=model.triple(h,r,t);n=model.triple(h[:,None].expand_as(neg),r[:,None].expand_as(neg),neg)
            if job['kind']=='TransE':
                # Author BaseModel uses mean negative distance before margin hinge.
                loss=F.relu(cfg['transe_margin']+p-n.mean(1)).mean()
            else:
                reg=model.entity(h).square().mean()+model.relation(r).square().mean()
                loss=F.softplus(-p).mean()+F.softplus(n).mean()+cfg['distmult_l2']*reg
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
        rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),elapsed_s=time.monotonic()-start)
        if epoch==1 or epoch%cfg['validation_every']==0 or epoch==cfg['epochs']:
            val=validate(model,selected,device);rec['validation']=val
            if val['raw_strict_mrr']>best:
                best=val['raw_strict_mrr'];chosen_epoch=epoch
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},kind=job['kind'],dim=job['dim'],seed=seed,epoch=epoch,nent=nent,nrel=nrel,train_only=True,selection='fixed old VALID raw MRR'),out/'best.pt')
            print('BASE_PROGRESS',job['id'],epoch,round(rec['train_loss'],6),val,round(rec['elapsed_s'],2),flush=True)
        records.append(rec)
        (out/'training-observations.json').write_text(json.dumps(records,indent=2))
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['state_dict'])
    # Save a small full-candidate validation replay fixture, never test answers.
    with torch.no_grad():
        fixture=selected[:64].to(device);scores=[]
        for part in fixture.split(8):scores.append(model.candidate_scores(part).cpu().numpy())
        scores=np.concatenate(scores);q=fixture.cpu().numpy();gold=scores[np.arange(len(q)),q[:,2]]
        np.savez_compressed(out/'valid-fixture.npz',queries=q,scores=scores,ranks=1+(scores>gold[:,None]).sum(1))
    for p,d in bound.items():assert sha(p)==d
    receipt=dict(id=job['id'],kind=job['kind'],seed=seed,dim=job['dim'],epochs_executed=cfg['epochs'],best_epoch=chosen_epoch,best_validation_mrr=best,
        entities=nent,relations=nrel,training_directed_queries=len(train),validation_selection_queries=len(selected),test_triples_read=False,new_snapshot_triples_read=False,
        elapsed_s=time.monotonic()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
        bound_input_sha256=bound,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},training_implementation_sha256=sha(Path(__file__)),
        runtime=dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),dtype=str(dtype)),
        limitations=['Common base preparation, not completed literature-method comparison.','Fixed validation subset of existing exposed development data; no new heldout claim.'])
    (out/'terminal-receipt.json').write_text(json.dumps(receipt,indent=2));print('BASE_TERMINAL',job['id'],chosen_epoch,best,flush=True)
