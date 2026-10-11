"""Same FP64 reduction tree with bounded cells and fewer narrow-column launches.

This is an optional implementation candidate, not a replacement for registered
scoring or evidence of method utility. Column tiles and reduction ordering match
shared_rank_cache_v2. Query tiles grow only when the entire column pool is small.
"""
import torch


@torch.no_grad()
def adaptive_fixed_scores(Q, E):
    assert Q.ndim == E.ndim == 2 and Q.shape[1] == E.shape[1]
    assert Q.dtype == E.dtype == torch.float64 and Q.device == E.device
    Q=Q.contiguous(); E=E.contiguous()
    output=torch.empty((len(Q),len(E)),dtype=torch.float64,device=Q.device)
    dimension=E.shape[1]; padded=1 << (dimension-1).bit_length()
    columns=min(256,max(1,len(E))); query_batch=max(64,(64*256)//columns)
    for off in range(0,len(Q),query_batch):
        part=Q[off:off+query_batch]
        for start in range(0,len(E),256):
            rows=E[start:start+256]
            assert len(part)*len(rows)<=64*256
            products=part[:,None,:]*rows[None,:,:]
            if padded>dimension:
                tree=torch.zeros((len(part),len(rows),padded),dtype=torch.float64,device=Q.device)
                tree[:,:,:dimension]=products
            else: tree=products
            while tree.shape[-1]>1: tree=tree[...,0::2]+tree[...,1::2]
            output[off:off+len(part),start:start+len(rows)]=tree[...,0]
    return output
