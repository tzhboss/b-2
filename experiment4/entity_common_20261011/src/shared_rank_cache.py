"""Query-grouped exact rank acceptance for shared entity patches.

All reference members of a query have max_rank=k. Their conjunction is
equivalent to rank of the current lowest-scoring reference <=k. Shared head,
gold and competitor changes are handled together. CUDA scoring uses a fixed
FP64 reduction kernel so full pools and subset columns have identical scores.
The cache is not an optimizer and does not guarantee new-knowledge utility.
"""
from dataclasses import dataclass
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _fixed_dot(Q, E, O, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
               P: tl.constexpr, K: tl.constexpr):
    pair = tl.program_id(0) * P + tl.arange(0, P)
    dim = tl.arange(0, K)
    h = pair // N; c = pair % N
    q = tl.load(Q + h[:, None] * D + dim[None, :],
                (pair[:, None] < M*N) & (dim[None, :] < D), other=0.)
    e = tl.load(E + c[:, None] * D + dim[None, :],
                (pair[:, None] < M*N) & (dim[None, :] < D), other=0.)
    value = tl.sum(q * e, axis=1)
    tl.store(O + pair, value, pair < M*N)


@torch.no_grad()
def fixed_scores(Q, E):
    assert Q.ndim == E.ndim == 2 and Q.shape[1] == E.shape[1]
    assert Q.dtype == E.dtype == torch.float64 and Q.device == E.device
    Q = Q.contiguous(); E = E.contiguous()
    if Q.is_cuda:
        output = torch.empty((len(Q), len(E)), dtype=torch.float64, device=Q.device)
        if output.numel():
            _fixed_dot[(triton.cdiv(output.numel(), 8),)](
                Q, E, output, len(Q), len(E), E.shape[1], 8,
                triton.next_power_of_2(E.shape[1]), num_warps=4,
                enable_fp_fusion=False)
        return output
    # CPU is an independent arithmetic path, not a bitwise GPU reference.
    output = torch.empty((len(Q), len(E)), dtype=torch.float64, device=Q.device)
    for off in range(0, len(Q), 16):
        output[off:off+16] = (Q[off:off+16, None, :] * E[None, :, :]).sum(2)
    return output


@dataclass
class RankPatch:
    version: int
    ids: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    head_queries: torch.Tensor
    head_directions: torch.Tensor
    head_scores: torch.Tensor
    thresholds: torch.Tensor
    counts: torch.Tensor
    passed: bool
    rescored_queries: int


class SharedRankCache:
    @torch.no_grad()
    def __init__(self, E, R, query_keys, references, k=10, score_batch=256):
        assert E.dtype == R.dtype == torch.float64 and E.device == R.device
        assert torch.isfinite(E).all() and torch.isfinite(R).all()
        self.E = E.clone(); self.R = R.clone(); self.device = E.device
        self.keys = torch.as_tensor(query_keys, dtype=torch.long, device=E.device).clone()
        self.references = torch.as_tensor(references, dtype=torch.long, device=E.device).clone()
        assert self.keys.ndim == self.references.ndim == 2
        assert self.keys.shape[1] == self.references.shape[1] == 2
        assert len(self.keys) and len(self.references) and len(torch.unique(self.keys,dim=0)) == len(self.keys)
        assert self.keys[:,0].min() >= 0 and self.keys[:,0].max() < len(E)
        assert self.keys[:,1].min() >= 0 and self.keys[:,1].max() < len(R)
        assert self.references[:,0].min() >= 0 and self.references[:,0].max() < len(self.keys)
        assert self.references[:,1].min() >= 0 and self.references[:,1].max() < len(E)
        assert len(torch.unique(self.references[:,0])) == len(self.keys)
        assert k >= 1 and score_batch >= 1
        self.k = int(k); self.batch = int(score_batch); self.version = 0
        self.Q = self.E[self.keys[:,0]] * self.R[self.keys[:,1]]
        self.S = torch.empty((len(self.keys),len(E)),dtype=E.dtype,device=E.device)
        for off in range(0,len(self.keys),self.batch):
            self.S[off:off+self.batch] = fixed_scores(self.Q[off:off+self.batch],self.E)
        values = self.S[self.references[:,0],self.references[:,1]]
        self.thresholds = self._minimum(values)
        self.counts = self._counts(self.S,self.thresholds)
        assert (self.counts < self.k).all(), 'Initial compiled reference scope is not retained'

    def _minimum(self, values):
        result = torch.full((len(self.keys),),float('inf'),dtype=torch.float64,device=self.device)
        result.scatter_reduce_(0,self.references[:,0],values,reduce='amin',include_self=True)
        assert torch.isfinite(result).all()
        return result

    def _counts(self, matrix, thresholds):
        return torch.cat([(matrix[off:off+self.batch] > thresholds[off:off+self.batch,None]).sum(1)
                          for off in range(0,len(matrix),self.batch)]) if len(matrix) else torch.empty(0,dtype=torch.long,device=self.device)

    @torch.no_grad()
    def propose(self, mutable_ids, new_rows):
        ids = torch.as_tensor(mutable_ids,dtype=torch.long,device=self.device).clone()
        rows = torch.as_tensor(new_rows,dtype=torch.float64,device=self.device).clone()
        assert ids.ndim == 1 and len(ids) and len(torch.unique(ids)) == len(ids)
        assert ids.min() >= 0 and ids.max() < len(self.E)
        assert rows.shape == self.E[ids].shape and torch.isfinite(rows).all()
        columns = fixed_scores(self.Q,rows)
        head_queries = torch.nonzero(torch.isin(self.keys[:,0],ids),as_tuple=False).flatten()
        if len(head_queries):
            updated = self.E.clone(); updated[ids] = rows
            head_directions = updated[self.keys[head_queries,0]] * self.R[self.keys[head_queries,1]]
            head_scores = fixed_scores(head_directions,updated)
            columns[head_queries] = head_scores[:,ids]
        else:
            head_directions = self.Q[:0].clone(); head_scores = self.S[:0].clone()
        reference_scores = self.S[self.references[:,0],self.references[:,1]].clone()
        if len(head_queries):
            query_map = torch.full((len(self.keys),),-1,dtype=torch.long,device=self.device)
            query_map[head_queries] = torch.arange(len(head_queries),device=self.device)
            changed_head_refs = query_map[self.references[:,0]] >= 0
            reference_scores[changed_head_refs] = head_scores[query_map[self.references[changed_head_refs,0]],self.references[changed_head_refs,1]]
        entity_map = torch.full((len(self.E),),-1,dtype=torch.long,device=self.device)
        entity_map[ids] = torch.arange(len(ids),device=self.device)
        changed_gold = entity_map[self.references[:,1]] >= 0
        reference_scores[changed_gold] = columns[self.references[changed_gold,0],entity_map[self.references[changed_gold,1]]]
        thresholds = self._minimum(reference_scores)
        counts = self.counts + (columns > self.thresholds[:,None]).sum(1) - (self.S[:,ids] > self.thresholds[:,None]).sum(1)
        rescore_mask = thresholds != self.thresholds
        rescore_mask[head_queries] = True
        affected = torch.nonzero(rescore_mask,as_tuple=False).flatten()
        for off in range(0,len(affected),self.batch):
            q = affected[off:off+self.batch]; s = self.S[q].clone(); s[:,ids] = columns[q]
            counts[q] = (s > thresholds[q,None]).sum(1)
        if len(head_queries):counts[head_queries] = self._counts(head_scores,thresholds[head_queries])
        return RankPatch(self.version,ids,rows,columns,head_queries,head_directions,head_scores,
                         thresholds,counts,bool((counts < self.k).all()),len(affected))

    @torch.no_grad()
    def commit(self, patch):
        assert patch.version == self.version, 'Stale proposal cannot be committed'
        if not patch.passed:return False
        assert (patch.counts < self.k).all()
        self.E[patch.ids] = patch.rows; self.S[:,patch.ids] = patch.columns
        self.Q[patch.head_queries] = patch.head_directions; self.S[patch.head_queries] = patch.head_scores
        self.thresholds = patch.thresholds.clone(); self.counts = patch.counts.clone()
        self.version += 1
        return True

    @torch.no_grad()
    def full_recount(self):
        """No cache-derived query/candidate scores: recompute the full shared pool."""
        Q = self.E[self.keys[:,0]] * self.R[self.keys[:,1]]
        assert torch.equal(Q,self.Q)
        fresh_counts = []
        for off in range(0,len(self.keys),self.batch):
            s = fixed_scores(Q[off:off+self.batch],self.E)
            assert torch.equal(s,self.S[off:off+self.batch]), 'Incremental score cache differs from full rescore'
            fresh_counts.append((s > self.thresholds[off:off+self.batch,None]).sum(1))
        values = self.S[self.references[:,0],self.references[:,1]]
        assert torch.equal(self._minimum(values),self.thresholds)
        assert torch.equal(torch.cat(fresh_counts),self.counts)
        assert (self.counts < self.k).all()
        return dict(query_keys=len(self.keys),references=len(self.references),commits=self.version,
                    exact_score_cache_replay=True,all_reference_members_retained=True)
