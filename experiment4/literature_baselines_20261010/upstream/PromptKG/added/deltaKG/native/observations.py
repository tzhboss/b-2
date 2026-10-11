"""Read-only observations around author evaluation; no additional model forward/RNG calls."""
import json
from pathlib import Path
from collections import defaultdict
import torch
import importlib
author = importlib.import_module('src.models.models')

def signature(f):
    return (tuple(f["input_ids"]), int(f["label"]), tuple(f["labels"]))

class ObservedCollator:
    def __init__(self, native, dataset, records, split):
        self.native = native
        self.split = split
        self.matches = defaultdict(list)
        if len(dataset) != len(records):
            raise RuntimeError("Source/features length mismatch; audit filtering before running")
        for f, r in zip(dataset.features, records):
            self.matches[signature(f)].append(r)

    def __call__(self, features):
        edit_ids = [self.matches[signature(f)] for f in features]
        n = self.native
        stable_ids = n.memory_perm[n.memory_pos:n.memory_pos+n.stable_batch_size]
        batch = n(features)
        batch["_observations"] = [
            {"role": "locality", "source_split": "stable", "source_row_candidates": [i]}
            for i in stable_ids
        ] + [
            {"role": "edit", "source_split": self.split, "source_row_candidates": ids}
            for ids in edit_ids
        ]
        return batch

class ObservedKGEditor(author.KGEditor):
    """Original _eval computes every forward and native rank unchanged."""
    def get_logits_orig_params_dict(self, batch):
        if not getattr(self, "_capture_eval", False):
            return super().get_logits_orig_params_dict(batch)
        def capture(module, args, output):
            self._before = output.logits.detach()
        hook = self.ex_model.register_forward_hook(capture)
        try:
            return super().get_logits_orig_params_dict(batch)
        finally:
            hook.remove()

    def _eval(self, batch, batch_idx):
        metadata = batch.pop("_observations", None)
        targets = batch["label"].detach().clone()
        input_ids = batch["input_ids"]
        original_factory = author.make_functional
        instance = self
        class ObservedFunctional:
            def __init__(self, native):
                self.native = native
            def eval(self):
                self.native.eval()
                return self
            def __call__(self, *args, **kwargs):
                result = self.native(*args, **kwargs)
                instance._after = result.logits.detach()
                return result
        def factory(model):
            return ObservedFunctional(original_factory(model))
        self._capture_eval = True
        author.make_functional = factory
        try:
            result = super()._eval(batch, batch_idx)
        finally:
            author.make_functional = original_factory
            self._capture_eval = False
        if getattr(self, "export_path", None):
            mask = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask[0]) != len(input_ids):
                raise RuntimeError("Expected exactly one mask per example")
            vectors = []
            ranks = []
            tops = []
            for full in [self._before, self._after]:
                score = full[mask[0], mask[1], self.entity_id_st:self.entity_id_ed]
                order = torch.sort(score, dim=1, descending=True)[1]
                inverse = torch.sort(order, dim=1)[1]
                rank = inverse[torch.arange(len(targets), device=targets.device), targets] + 1
                ranks.append(rank.cpu().tolist())
                tops.append(order[:, :3].cpu().tolist())
                vectors.append(score[torch.arange(len(targets), device=targets.device), targets].cpu().tolist())
            if ranks[1] != result["ranks"].tolist():
                raise RuntimeError("Observation ranks differ from native rank output")
            with Path(self.export_path).open("a") as handle:
                for i, target in enumerate(targets.cpu().tolist()):
                    info = metadata[i] if metadata else {"role": "locality" if i < len(targets)-self.edit_num else "edit", "source_row_candidates": []}
                    handle.write(json.dumps(dict(
                        info, batch_idx=batch_idx, position=i, target_entity_index=target,
                        rank_before=ranks[0][i], rank_after=ranks[1][i],
                        target_logit_before=vectors[0][i], target_logit_after=vectors[1][i],
                        top3_before=tops[0][i], top3_after=tops[1][i],
                        masked_input_ids=input_ids[i].cpu().tolist(),
                        evaluation_label=self.evaluation_label)) + "\n")
        del self._before, self._after
        return result

def summarize(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    out = {"row_count": len(rows)}
    for role in ["edit", "locality"]:
        r = [x for x in rows if x["role"] == role]
        if not r:
            raise RuntimeError("No examples for " + role)
        mean_before = sum(x["rank_before"] for x in r) / len(r)
        mean_after = sum(x["rank_after"] for x in r) / len(r)
        v = {"count": len(r), "mean_rank_before": mean_before, "mean_rank_after": mean_after,
             "mrr_after": sum(1/x["rank_after"] for x in r)/len(r),
             "unique_source_rows": len(set(i for x in r for i in x["source_row_candidates"]))}
        for k in [1,3,5]:
            v["hits"+str(k)+"_after"] = sum(x["rank_after"] <= k for x in r)/len(r)
        if role == "edit":
            v["article_ER_roc"] = abs(mean_after-mean_before)/mean_before
        else:
            denom = sum(x["rank_before"] <= 3 for x in r)
            v["article_RK_at3"] = sum(x["rank_after"] <= 3 for x in r)/denom if denom else None
            v["article_RK_roc"] = abs(mean_after-mean_before)/mean_after
            v["baseline_hits3_count"] = denom
        out[role] = v
    out["aggregation_scope"] = "Repeated native batch exposures, not unique full L-Test; rank-rate changes are not ROC AUC"
    return out
