"""CPU-only native data/model API preflight; no optimizer steps or reported KGE metrics."""
import json, os, pathlib, sys, time
os.environ['CUDA_VISIBLE_DEVICES']=''; sys.dont_write_bytecode=True
base=pathlib.Path(__file__).resolve().parent
c=json.loads((base/'config.json').read_text()); repo=(base/c['repo_root']).resolve()
sys.path[:0]=[str(base/c['vendor']),str(repo)]; os.chdir(repo)
sys.argv=c['command']
import torch
from src.parse_args import args
from src.utils import set_seeds
from src.data_load.KnowledgeGraph import KnowledgeGraph
from src.data_load.data_loader import TrainDatasetMarginLoss
from src.model.LoraKGE_Layers import TransE
torch.set_num_threads(4); set_seeds(args.random_seed)
args.device=torch.device('cpu'); args.data_path='./data/'+args.dataset+'/'
args.snapshot=args.snapshot_test=args.snapshot_valid=0
args.valid=False; args.test_FWT=False
start=time.monotonic(); kg=KnowledgeGraph(args); model=TransE(args,kg)
rows=[]
for i in range(5):
    args.snapshot=args.snapshot_test=args.snapshot_valid=i
    ds=TrainDatasetMarginLoss(args,kg)
    bh,br,bt,by=ds.collate_fn([ds[0],ds[1]])
    model.train(); loss=model.loss(bh,br,bt,by); assert torch.isfinite(loss)
    loss.backward(); model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(): pred=model.predict(bh[:2],br[:2],stage='Test')
    assert pred.shape==(2,kg.snapshots[i].num_ent) and torch.isfinite(pred).all()
    rows.append({'snapshot':i,'entities':kg.snapshots[i].num_ent,'relations_including_inverse':kg.snapshots[i].num_rel,'train_triples':len(kg.snapshots[i].train),'valid_triples':len(kg.snapshots[i].valid),'test_triples':len(kg.snapshots[i].test),'predict_shape':list(pred.shape),'entity_ranks':getattr(args,'ent_r_list',None)})
    if i<4: model.switch_snapshot()
out={'kind':'CPU_API_PREFLIGHT_NOT_TRAINING_OR_REPRODUCTION','optimizer_steps':0,'elapsed_seconds':time.monotonic()-start,'snapshots':rows}
(base/'preflight_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
