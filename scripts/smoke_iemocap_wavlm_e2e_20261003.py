#!/usr/bin/env python3
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import Wav2Vec2FeatureExtractor
from run_iemocap_wavlm_e2e_within_aux_shard_20261003 import (
    Model, speaker_id, session_id, equal_speaker_weights, batch_inputs, wmse, stable_seed
)

CFG=Path("/data/lc/tzh/configs/experiments/EXP-20261003-07.yaml")

def main():
    cfg=yaml.safe_load(CFG.read_text())
    held="Ses01"; seed=int(cfg["seeds"][0])
    cols=["file","EmoAct","EmoVal","EmoDom"]
    frames=[pd.read_parquet(f,columns=cols) for f in sorted(glob.glob(cfg["dataset_glob"]))]
    d=pd.concat(frames,ignore_index=True).dropna().copy()
    d["speaker_id"]=d.file.map(speaker_id); d["session_id"]=d.file.map(session_id)
    d=d.rename(columns={"EmoAct":"arousal","EmoVal":"valence","EmoDom":"dominance"})
    d["audio_path"]=d.file.map(lambda x:str(Path(cfg["audio_root"])/str(x)))
    tr=d.session_id.ne(held).to_numpy()
    trd=d.loc[tr].copy().reset_index(drop=True)
    trsp=trd.speaker_id.astype(str).to_numpy()
    Y=trd[["arousal","valence","dominance"]].to_numpy(np.float32)
    sw=equal_speaker_weights(trsp)
    ymean=np.average(Y,axis=0,weights=sw).astype(np.float32)
    yvar=np.average((Y-ymean)**2,axis=0,weights=sw).astype(np.float32)
    ystd=np.sqrt(np.maximum(yvar,1e-6)).astype(np.float32)
    yz=((Y-ymean)/ystd).astype(np.float32)
    b=np.empty_like(yz)
    for sp in sorted(set(trsp)):
        m=trsp==sp; b[m]=yz[m].mean(0)
    w=yz-b
    sizes=np.array([Path(p).stat().st_size for p in trd.audio_path.astype(str)])
    bs=int(cfg["batch_size"])
    ids=np.argsort(sizes)[-bs:]
    paths=trd.iloc[ids].audio_path.astype(str).tolist()
    device=torch.device("cuda:0")
    fe=Wav2Vec2FeatureExtractor.from_pretrained(cfg["model_dir"],local_files_only=True)
    init_seed=stable_seed(seed,0,"smoke")
    torch.manual_seed(init_seed); torch.cuda.manual_seed_all(init_seed)
    model=Model(cfg["model_dir"],cfg["hidden_dims"],float(cfg["dropout"]),True).to(device).train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(cfg["encoder_lr"]))
    iv,am=batch_inputs(fe,paths,device)
    yb=torch.from_numpy(yz[ids]).to(device); wb=torch.from_numpy(w[ids]).to(device); swb=torch.from_numpy(sw[ids]).to(device)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        o,ww=model(iv,am)
        loss=wmse(o,yb,swb)+float(cfg["aux_weight"])*wmse(ww,wb,swb)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    peak=torch.cuda.max_memory_allocated()/1024**3
    print("SMOKE_OK","batch",bs,"loss",float(loss.detach().cpu()),"peak_GiB",round(peak,3),"files",[Path(p).name for p in paths],flush=True)

if __name__=="__main__":
    main()
