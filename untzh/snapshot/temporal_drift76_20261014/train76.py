"""Simple chronological temporal bilinear baselines; no novel loss or protection."""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prepare76 import write_json

ROOT = Path(__file__).resolve().parent
CONFIG = {'seed':17,'embedding_width':128,'initial_epochs':50,'increment_epochs':15,
          'batch_size':1024,'optimizer':'Adagrad','lr':0.1,'n3':0.001,
          'replay_original_fact_memory':4096,'replay_batch_ratio':0.25,
          'time_smoothness':0,'train_candidate_scope':'entities seen in TRAIN up to current year',
          'first_year':2005,'last_year':2015}


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class TemporalBilinear(nn.Module):
    def __init__(self, architecture, ne, nr, nt):
        super().__init__()
        self.architecture = architecture
        self.E = nn.Parameter(torch.randn(ne,128,device='cuda')*0.1)
        self.R = nn.Parameter(torch.randn(nr*2,128,device='cuda')*0.1)
        t = torch.ones(nt,128,device='cuda')
        if architecture == 'TComplEx':
            t[:,64:] = 0.01*torch.randn(nt,64,device='cuda')
        else:
            t += 0.01*torch.randn_like(t)
        self.T = nn.Parameter(t)

    def query_and_factors(self, x):
        h,r,t,d = x.unbind(1)
        eh,rr,tt,et = self.E[h],self.R[r],self.T[d],self.E[t]
        if self.architecture == 'TDistMult':
            rt = rr*tt
            return eh*rt, (eh.abs(),rt.abs(),et.abs())
        hr,hi = eh.chunk(2,1);rrr,rri = rr.chunk(2,1);tr,ti = tt.chunk(2,1)
        rr_eff,ri_eff = rrr*tr-rri*ti, rri*tr+rrr*ti
        q = torch.cat([hr*rr_eff-hi*ri_eff,hi*rr_eff+hr*ri_eff],1)
        er,ei = et.chunk(2,1)
        factors = (torch.sqrt(hr.square()+hi.square()+1e-12),
                   torch.sqrt(rr_eff.square()+ri_eff.square()+1e-12),
                   torch.sqrt(er.square()+ei.square()+1e-12))
        return q,factors

    def query(self, x):
        return self.query_and_factors(x)[0]

    def loss(self, x, active_n):
        q,factors = self.query_and_factors(x)
        logits = q@self.E[:active_n].T
        ce = F.cross_entropy(logits,x[:,2])
        reg = CONFIG['n3']*sum(f.pow(3).sum() for f in factors)/len(x)
        return ce+reg,ce,reg


def optimizer(model):
    return torch.optim.Adagrad(model.parameters(), lr=CONFIG['lr'])


def checkpoint(model,opt,path,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'architecture':model.architecture,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':opt.state_dict(),'metadata':metadata},path)


def fit(model,opt,train,year,active_n,epochs,arm,memory=None,freeze=None):
    start = time.perf_counter()
    rng = np.random.default_rng(17+year)
    x = torch.as_tensor(train,device='cuda')
    old = None if memory is None or not len(memory) else torch.as_tensor(memory,device='cuda')
    epoch_log=[]
    for ep in range(epochs):
        permutation=rng.permutation(len(train))
        total_loss=total_ce=total_reg=0.; seen=0; replay_examples=0
        for b in range(0,len(train),CONFIG['batch_size']):
            batch=x[torch.as_tensor(permutation[b:b+CONFIG['batch_size']],device='cuda')]
            loss,ce,reg=model.loss(batch,active_n)
            if old is not None:
                m=max(1,len(batch)//4)
                replay=old[torch.as_tensor(rng.integers(0,len(old),size=m),device='cuda')]
                lp,cp,rp=model.loss(replay,active_n)
                loss=(loss+0.25*lp)/1.25;ce=(ce+0.25*cp)/1.25;reg=(reg+0.25*rp)/1.25
                replay_examples+=m
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if freeze is not None:
                ne,nr,nt = freeze
                model.E.grad[:ne]=0
                model.R.grad[:nr]=0
                model.R.grad[DATA_NR:DATA_NR+nr]=0
                model.T.grad[:nt]=0
            opt.step()
            n=len(batch);seen+=n
            total_loss+=float(loss.detach())*n;total_ce+=float(ce.detach())*n;total_reg+=float(reg.detach())*n
        row={'arm':arm,'year':year,'epoch':ep+1,'loss':total_loss/seen,'ce':total_ce/seen,
             'regularization':total_reg/seen,'directional_current_examples':seen,
             'replay_directional_examples':replay_examples,'elapsed':time.perf_counter()-start}
        assert np.isfinite(row['loss'])
        epoch_log.append(row)
        write_json(ROOT/'status.json',{'stage':'train',**row,'pid':os.getpid()})
        if ep == 0 or (ep+1)%5==0 or ep+1==epochs:
            print('EPOCH',json.dumps(row),flush=True)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    (ROOT/'training').mkdir(exist_ok=True)
    write_json(ROOT/f'training/{arm}-{year}.json',{'epochs':epoch_log,'seconds':seconds})
    return seconds


def memory_for_year(data, year):
    # A reproducible uniform memory over original historical TRAIN rows, sampled before seeing this year's TRAIN.
    original=data['train'][:len(data['train'])//2]
    past=original[data['dyear'][original[:,3]]<year]
    rng=np.random.default_rng(170000+year)
    chosen=np.sort(rng.choice(len(past),min(CONFIG['replay_original_fact_memory'],len(past)),replace=False))
    orig=past[chosen]
    inv=orig[:,[2,1,0,3]].copy();inv[:,1]+=DATA_NR
    (ROOT/'memory').mkdir(exist_ok=True)
    np.save(ROOT/f'memory/{year}-original-train.npy',orig)
    return np.concatenate([orig,inv])


DATA_NR=251


def main():
    global DATA_NR
    setup();start=time.perf_counter()
    data=np.load(ROOT/'prepared/data.npz')
    stats=json.loads((ROOT/'data-statistics.json').read_text())
    DATA_NR=stats['relations']
    write_json(ROOT/'config.json',CONFIG)
    costs={'training':{},'notes':['Shared 2005 fit charged once per architecture; branches reuse checkpoint and optimizer.',
                                'Replay adds historical batches; compute and memory logged separately.']}
    for architecture in ('TDistMult','TComplEx'):
        torch.manual_seed(17)
        model=TemporalBilinear(architecture,stats['entities'],DATA_NR,stats['dates'])
        opt=optimizer(model)
        year=2005
        train=data['train'][data['dyear'][data['train'][:,3]]==year]
        active_n=int((data['eyear']<=year).sum())
        sec=fit(model,opt,train,year,active_n,CONFIG['initial_epochs'],architecture+'_shared')
        checkpoint(model,opt,ROOT/f'models/{architecture}_shared/2005.pt',{'year':year,'active_n':active_n,'config':CONFIG})
        costs['training'][architecture+'_shared-2005']=sec
        base_state=copy.deepcopy(model.state_dict());base_opt=copy.deepcopy(opt.state_dict())
        modes=['finetune','replay']+(['frozen_history'] if architecture=='TComplEx' else [])
        for mode in modes:
            arm=architecture+'_'+mode
            model.load_state_dict(base_state);opt=optimizer(model);opt.load_state_dict(copy.deepcopy(base_opt))
            for year in range(2006,2016):
                oldn=int((data['eyear']<year).sum());active_n=int((data['eyear']<=year).sum())
                oldrn=int((data['ryear']<year).sum());oldt=int((data['dyear']<year).sum())
                train=data['train'][data['dyear'][data['train'][:,3]]==year]
                frozen=mode=='frozen_history'
                previous=(model.E[:oldn].detach().clone(),model.R.detach().clone(),model.T[:oldt].detach().clone()) if frozen else None
                oldtime=model.T[:oldt].detach().clone()
                memory=memory_for_year(data,year) if mode=='replay' else None
                sec=fit(model,opt,train,year,active_n,CONFIG['increment_epochs'],arm,memory,
                        (oldn,oldrn,oldt) if frozen else None)
                metadata={'year':year,'active_n':active_n,'config':CONFIG,
                          'historical_time_rows_equal':bool(torch.equal(oldtime,model.T[:oldt])),
                          'historical_time_rows_max_abs_change':float((oldtime-model.T[:oldt]).abs().max())}
                if frozen:
                    metadata['old_entity_rows_equal']=bool(torch.equal(previous[0],model.E[:oldn]))
                    metadata['old_relation_rows_equal']=bool(torch.equal(previous[1][:oldrn],model.R[:oldrn]) and torch.equal(previous[1][DATA_NR:DATA_NR+oldrn],model.R[DATA_NR:DATA_NR+oldrn]))
                    assert metadata['old_entity_rows_equal'] and metadata['old_relation_rows_equal'] and metadata['historical_time_rows_equal']
                elif mode=='finetune':
                    assert metadata['historical_time_rows_equal']
                checkpoint(model,opt,ROOT/f'models/{arm}/{year}.pt',metadata)
                costs['training'][f'{arm}-{year}']=sec
                write_json(ROOT/'training-costs.json',costs)
                print('SNAPSHOT_COMPLETE',arm,year,sec,metadata['historical_time_rows_equal'],flush=True)
    costs['train_total_wall_seconds']=time.perf_counter()-start
    write_json(ROOT/'training-costs.json',costs)
    write_json(ROOT/'status.json',{'stage':'trained','pid':os.getpid(),'seconds':costs['train_total_wall_seconds']})
    print('TRAIN_COMPLETE',costs['train_total_wall_seconds'],flush=True)


if __name__=='__main__':
    main()
