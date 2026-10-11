"""CPU tensor parity for actual author _eval on a tiny EXBert/hypernetwork."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import torch
from transformers import BertConfig

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
author = importlib.import_module("src.models.models")
obs = importlib.import_module("observations")
runner = importlib.import_module("run_native")

torch.set_num_threads(1)
torch.manual_seed(127)
cfg = BertConfig(vocab_size=48,hidden_size=16,intermediate_size=32,
                 num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=32,
                 hidden_dropout_prob=0.0,attention_probs_dropout_prob=0.0)
cfg.ex_size = 4
cfg.kb_layer = [1]

class TinyInit:
    def __init__(self):
        author.BaseKEModule.__init__(self)
        self.ex_model = author.EXBertForMaskedLM(cfg).eval()
        names = {n for n,p in self.ex_model.named_parameters()
                 if ("dense_in_ex" in n or "dense_out_ex" in n) and "bias" not in n}
        self.ex_learner = author.OneShotLearner(
            self.ex_model,vocab_dim=48,embedding_dim=16,hidden_dim=8,
            condition_dim=16,include_set=names,max_scale=.01,
            embedding_init=self.ex_model.bert.embeddings.word_embeddings.weight.data)
        self.tokenizer = types.SimpleNamespace(mask_token_id=3)
        self.entity_id_st = 32
        self.entity_id_ed = 42
        self.edit_num = 2
        self.eval()

    def get_logits_orig_params_dict(self,batch):
        logits,params = super().get_logits_orig_params_dict(batch)
        self.observed_shift = {k:v.detach().clone() for k,v in params.items()}
        return logits,params

class NativeTiny(TinyInit,author.KGEditor):
    pass

class ObservedTiny(TinyInit,obs.ObservedKGEditor):
    pass

native = NativeTiny()
observed = ObservedTiny()
observed.load_state_dict(native.state_dict(),strict=True)
input_ids = torch.tensor([[1,3,5,6,2],[1,7,3,8,2],[1,3,9,10,2],[1,11,3,12,2]])
batch = dict(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),
             cond_input_ids=torch.tensor([[1,8,9,2],[1,9,10,2],[1,10,11,2],[1,11,12,2]]),
             cond_attention_mask=torch.ones(4,4,dtype=torch.long),
             label=torch.tensor([0,1,2,3]),labels=torch.tensor([0,1,4,5]))
before_states = {k:v.clone() for k,v in native.state_dict().items()}
start_rng = torch.get_rng_state().clone()
native_capture = {}
def before_hook(module,args,output):
    if "before" not in native_capture:
        native_capture["before"] = output.logits.detach().clone()
hook = native.ex_model.register_forward_hook(before_hook)
original_factory = author.make_functional
class CaptureFunctional:
    def __init__(self,model):
        self.model=model
    def eval(self):
        self.model.eval()
        return self
    def __call__(self,*args,**kwargs):
        result=self.model(*args,**kwargs)
        native_capture["after"]=result.logits.detach().clone()
        return result
def native_factory(model):
    return CaptureFunctional(original_factory(model))
author.make_functional=native_factory
try:
    with torch.no_grad():
        expected = native._eval(copy.deepcopy(batch),0)
finally:
    hook.remove()
    author.make_functional=original_factory
native_rng = torch.get_rng_state().clone()
torch.set_rng_state(start_rng)
with tempfile.TemporaryDirectory() as temp:
    export = Path(temp)/"examples.jsonl"
    observed.export_path = str(export)
    observed.evaluation_label = "synthetic CPU fixture only"
    b = copy.deepcopy(batch)
    b["_observations"] = [
        dict(role="locality" if i<2 else "edit",source_split="fixture",
             source_row_candidates=[i]) for i in range(4)]
    with torch.no_grad():
        actual = observed._eval(b,0)
    rows = [json.loads(s) for s in export.read_text().splitlines()]
    aggregate = obs.summarize(export)
    assert len(rows)==4
    assert [r["rank_after"] for r in rows] == expected["ranks"].tolist()
    assert all(len(r["top3_after"])==3 for r in rows)
    mask=(input_ids==3).nonzero(as_tuple=True)
    for when in ["before","after"]:
        scores=native_capture[when][mask[0],mask[1],32:42]
        order=torch.sort(scores,dim=1,descending=True)[1]
        inverse=torch.sort(order,dim=1)[1]
        ranks=inverse[torch.arange(4),batch["label"]]+1
        for i,row in enumerate(rows):
            assert row["rank_"+when]==int(ranks[i])
            assert row["top3_"+when]==order[i,:3].tolist()
            assert row["target_logit_"+when]==float(scores[i,batch["label"][i]])
    assert aggregate["edit"]["count"]==2 and aggregate["locality"]["count"]==2
for key,value in expected.items():
    if isinstance(value,np.ndarray):
        np.testing.assert_array_equal(value,actual[key])
    elif torch.is_tensor(value):
        torch.testing.assert_close(value,actual[key],rtol=0,atol=0)
    else:
        assert value==actual[key]
assert torch.equal(native_rng,torch.get_rng_state())
for name,weight in before_states.items():
    torch.testing.assert_close(weight,native.state_dict()[name],rtol=0,atol=0)
    torch.testing.assert_close(weight,observed.state_dict()[name],rtol=0,atol=0)
for name,shift in native.observed_shift.items():
    torch.testing.assert_close(shift,observed.observed_shift[name],rtol=0,atol=0)
# Import and parser path exercised without trainer/model loading or GPU operations.
parser = runner.native_parser()
parsed = parser.parse_args(["--data_dir","fixture","--dirpath","fixture","--max_epochs","20",
                           "--num_workers","4","--gpus","1","--accelerator","ddp"])
assert parsed.max_epochs==20 and parsed.num_workers==4
print(json.dumps(dict(status="CPU_TENSOR_PARITY_PASS",torch_version=torch.__version__,
    device=str(next(native.parameters()).device),native_ranks=expected["ranks"].tolist(),
    native_return_fields=list(expected),edit_shift_matrices=len(native.observed_shift),
    original_weights_unchanged=True,rng_unchanged=True,export_rows=4,
    adapter_imports_and_native_parser=True,before_after_export_tensor_values_equal=True,training_executed=False)))
