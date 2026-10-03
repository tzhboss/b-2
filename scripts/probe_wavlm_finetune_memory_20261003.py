#!/usr/bin/env python3
import argparse, json
import torch
import torch.nn as nn
from transformers import WavLMModel

ap=argparse.ArgumentParser()
ap.add_argument("--model-dir",required=True)
ap.add_argument("--seconds",type=float,required=True)
ap.add_argument("--device",default="cuda:0")
args=ap.parse_args()

device=torch.device(args.device)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
model=WavLMModel.from_pretrained(args.model_dir,local_files_only=True).to(device)
model.feature_extractor._freeze_parameters()
model.gradient_checkpointing_enable()
head=nn.Linear(model.config.hidden_size,3).to(device)
opt=torch.optim.AdamW([
    {"params":[p for p in model.parameters() if p.requires_grad],"lr":1e-5},
    {"params":head.parameters(),"lr":1e-4}
],weight_decay=1e-4)
n=int(round(args.seconds*16000))
x=torch.randn(1,n,device=device)
mask=torch.ones(1,n,dtype=torch.long,device=device)
target=torch.zeros(1,3,device=device)
with torch.autocast(device_type="cuda",dtype=torch.float16):
    out=model(input_values=x,attention_mask=mask)
    h=out.last_hidden_state
    fm=model._get_feature_vector_attention_mask(h.shape[1],mask).to(h.dtype)
    pooled=(h*fm.unsqueeze(-1)).sum(1)/fm.sum(1,keepdim=True).clamp_min(1)
    pred=head(pooled)
    loss=(pred-target).pow(2).mean()
loss.backward()
torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.0)
opt.step(); opt.zero_grad(set_to_none=True)
print(json.dumps({
  "seconds":args.seconds,
  "samples":n,
  "loss":float(loss.detach().cpu()),
  "peak_allocated_gb":torch.cuda.max_memory_allocated(device)/2**30,
  "peak_reserved_gb":torch.cuda.max_memory_reserved(device)/2**30
},indent=2))
