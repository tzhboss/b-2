#!/usr/bin/env python3
import argparse,sys,torch
sys.path.insert(0,'/data/lc/tzh/scripts')
from run_iemocap_wavlm_e2e_within_aux_shard_20261003 import Model,load_audio
from transformers import Wav2Vec2FeatureExtractor

ap=argparse.ArgumentParser()
ap.add_argument('--scaler',type=int,default=1)
ap.add_argument('--optimizer',type=int,default=1)
ap.add_argument('--nonreentrant',type=int,default=0)
a=ap.parse_args()

dev=torch.device('cuda:0')
m=Model('/data/lc/models/microsoft-wavlm-large',[128,64],0.1,False)
if a.nonreentrant:
    m.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
else:
    m.wavlm.gradient_checkpointing_enable()
m.wavlm=m.wavlm.to(dev); m.trunk=m.trunk.to(dev); m.overall=m.overall.to(dev); m.within=m.within.to(dev)
print('MODEL_READY',flush=True)
enc=[p for n,p in m.named_parameters() if n.startswith('wavlm.') and p.requires_grad]
head=[p for n,p in m.named_parameters() if not n.startswith('wavlm.') and p.requires_grad]
opt=torch.optim.AdamW([{'params':enc,'lr':1e-5},{'params':head,'lr':1e-4}],weight_decay=1e-4) if a.optimizer else None
scaler=torch.amp.GradScaler('cuda',enabled=bool(a.scaler))

p='/data/lc/dataset/iemocap/data/IEMOCAP/Ses04F_impro07_F079.wav'
y=load_audio(p,max_seconds=10.0)
fe=Wav2Vec2FeatureExtractor.from_pretrained('/data/lc/models/microsoft-wavlm-large',local_files_only=True)
x=fe([y],sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors='pt')
iv=x.input_values.to(dev); am=x.attention_mask.to(dev)
target=torch.zeros(1,3,device=dev)
with torch.autocast(device_type='cuda',dtype=torch.float16):
    o,w=m(iv,am)
    loss=((o-target)**2).mean()
print('FORWARD',float(loss.detach()),flush=True)
if a.scaler:
    scaler.scale(loss).backward()
else:
    loss.backward()
print('BACKWARD_OK',flush=True)
checks={}
for name in ['wavlm.encoder.layers.0.attention.q_proj.weight','wavlm.encoder.layers.23.attention.q_proj.weight','overall.weight']:
    p0=dict(m.named_parameters())[name]
    checks[name]=None if p0.grad is None else float(p0.grad.abs().mean().cpu())
print(checks,flush=True)
if opt:
    if a.scaler:
        scaler.unscale_(opt); scaler.step(opt); scaler.update()
    else:
        opt.step()
    print('STEP_OK',flush=True)
