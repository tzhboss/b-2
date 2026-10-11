"""Experiment4 phase24: independently train real DistMult/ComplEx KGE models.

FB15k-237 TRAIN alone for model fitting. VALID only for periodic model
diagnostics, no TEST loading. One training seed per model unless requested.
Shared entity representations used for both heads and tails.
"""
import argparse, json, time, hashlib, random, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load(root):
    def table(name):
        return {s:int(i) for i,s in (line.split('\t',1) for line in (root/name).read_text(encoding="utf-8").splitlines())}
    E=table('entities.dict'); R=table('relations.dict')
    def triples(name):
        return torch.tensor([[E[h],R[r],E[t]] for h,r,t in
           (line.split('\t') for line in (root/name).read_text(encoding='utf-8').splitlines())],dtype=torch.long)
    return E,R,triples('train.txt'),triples('valid.txt')

class KGModel(nn.Module):
    def __init__(self,nent,nrel,d,complex_model):
        super().__init__()
        self.kind="ComplEx" if complex_model else "DistMult"
        self.e_re=nn.Embedding(nent,d)
        self.r_re=nn.Embedding(nrel,d)
        nn.init.normal_(self.e_re.weight,std=.12)
        nn.init.normal_(self.r_re.weight,std=.12)
        if complex_model:
            self.e_im=nn.Embedding(nent,d)
            self.r_im=nn.Embedding(nrel,d)
            nn.init.normal_(self.e_im.weight,std=.12)
            nn.init.normal_(self.r_im.weight,std=.12)
        else:
            self.e_im=None;self.r_im=None
    def score(self,h,r,t):
        h0=self.e_re(h);r0=self.r_re(r);t0=self.e_re(t)
        if self.e_im is None: return (h0*r0*t0).sum(-1)
        hi=self.e_im(h);ri=self.r_im(r);ti=self.e_im(t)
        qr=h0*r0-hi*ri
        qi=h0*ri+hi*r0
        return (qr*t0+qi*ti).sum(-1)
    def tail_scores(self,h,r):
        h0=self.e_re(h);r0=self.r_re(r)
        if self.e_im is None:return (h0*r0)@self.e_re.weight.T
        hi=self.e_im(h);ri=self.r_im(r)
        return (h0*r0-hi*ri)@self.e_re.weight.T+(h0*ri+hi*r0)@self.e_im.weight.T

@torch.no_grad()
def eval_valid(model,train,valid,n=512,seed=124):
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(valid),size=min(n,len(valid)),replace=False)
    arr=valid[indices]
    true=defaultdict(set)
    for h,r,t in torch.cat((train,valid)).tolist():true[(h,r)].add(t)
    mr=[];hits=[]
    for batch in arr.split(32):
        batch=batch.to(next(model.parameters()).device)
        scores=model.tail_scores(batch[:,0],batch[:,1]).float().cpu().numpy()
        for i,(h,r,t) in enumerate(batch.cpu().tolist()):
            for x in true[(h,r)]:
                if x!=t:scores[i,x]=-np.inf
            actual=scores[i,t]
            rank=1+int(np.count_nonzero(scores[i]>actual))
            mr.append(1/rank);hits.append(int(rank<=10))
    return dict(filtered_valid_MRR=float(np.mean(mr)),filtered_valid_Hits10=float(np.mean(hits)),
                n=len(mr),filtered_known_scope="TRAIN_PLUS_VALID_ONLY_NOT_STANDARD_TEST")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--kind',choices=['DistMult','ComplEx'],required=True)
    ap.add_argument('--seed',type=int,default=401)
    ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--dim',type=int,default=64)
    ap.add_argument('--batch',type=int,default=4096)
    ap.add_argument('--lr',type=float,default=.007)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    E,R,train,valid=load(args.dataset)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model=KGModel(len(E),len(R),args.dim,args.kind=='ComplEx').to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=torch.Generator().manual_seed(args.seed)
    run={"kind":args.kind,"seed":args.seed,"dim":args.dim,"epochs_requested":args.epochs,
         "n_ent":len(E),"n_rel":len(R),"n_train":len(train),"n_valid":len(valid),
         "device":str(device),"learning_rate":args.lr,"batch":args.batch,
         "neg_per_pos":4,"train_only":True,"TEST_touched":False,
         "dataset_sha256":{name:hashlib.sha256((args.dataset/name).read_bytes()).hexdigest()
                            for name in ['train.txt','valid.txt','entities.dict','relations.dict']},
         "epochs":[]}
    start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();means=[]
        for indices in torch.randperm(len(train),generator=rng).split(args.batch):
            pos=train[indices].to(device);n=len(pos)
            neg=torch.randint(len(E),(n,4),generator=rng).to(device)
            h=pos[:,0];r=pos[:,1];t=pos[:,2]
            p=model.score(h,r,t)
            negsc=model.score(h[:,None].expand(-1,4),r[:,None].expand(-1,4),neg)
            reg=(model.e_re(h).square().mean()+model.r_re(r).square().mean())
            if model.e_im is not None:
                reg+=model.e_im(h).square().mean()+model.r_im(r).square().mean()
            loss=F.softplus(-p).mean()+F.softplus(negsc).mean()+.0005*reg
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            means.append(float(loss.item()))
        rec={"epoch":epoch,"train_loss":float(np.mean(means)),"elapsed_s":round(time.perf_counter()-start,2)}
        if epoch in (1,5,10,15,20,25,args.epochs):
            model.eval();rec["valid_diagnostic"]=eval_valid(model,train,valid)
        run["epochs"].append(rec)
        print("EPOCH",args.kind,args.seed,epoch, "LOSS",round(rec["train_loss"],4),
             "VALID",rec.get("valid_diagnostic",{}),"S",rec["elapsed_s"],flush=True)
        checkpoint={"kind":args.kind,"seed":args.seed,"epoch":epoch,"dim":args.dim,
                    "entities":model.e_re.weight.detach().cpu(),
                    "relations":model.r_re.weight.detach().cpu()}
        if model.e_im is not None:
            checkpoint["entities_im"]=model.e_im.weight.detach().cpu()
            checkpoint["relations_im"]=model.r_im.weight.detach().cpu()
        # Save each milestone for comparison and recovery.
        if epoch in (5,10,15,20,25,args.epochs):
            f=args.out/f'{args.kind.lower()}_seed{args.seed}_epoch{epoch}.pt'
            torch.save(checkpoint,f)
            rec["checkpoint_sha256"]=hashlib.sha256(f.read_bytes()).hexdigest()
            rec["checkpoint_file"]=f.name
        (args.out/f'{args.kind.lower()}_seed{args.seed}_training.json').write_text(
            json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    print("TRAIN_FINISHED",args.kind,args.seed,"ELAPSED",round(time.perf_counter()-start,2),flush=True)

if __name__=="__main__":main()
