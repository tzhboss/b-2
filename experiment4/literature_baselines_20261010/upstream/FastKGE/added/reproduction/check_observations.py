"""Synthetic CPU check of unchanged native ranking and added observation files."""
import os,sys,pathlib,json,tempfile,types
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
base=pathlib.Path(__file__).resolve().parent
c=json.loads((base/'config.json').read_text());repo=(base/c['repo_root']).resolve()
sys.path[:0]=[str(base/c['vendor']),str(repo)];os.chdir(repo);sys.argv=c['command']
import torch,numpy as np
from src.model.model_process import DevBatchProcessor
import main
scores=torch.tensor([[.1,.8,.6,.3],[.2,.9,.5,.4],[.1,.8,.7,.4]])
head=torch.tensor([0,1,2]);rel=torch.tensor([0,0,1]);tail=torch.tensor([1,2,3]);labels=torch.zeros(3,4);labels[1,1]=1
class Model:
    lora_ent_embeddings_list=None
    def eval(self):pass
    def predict(self,h,r,stage):return scores.clone()
with tempfile.TemporaryDirectory(dir=base) as td:
    observation=pathlib.Path(td)/'observations'
    args=types.SimpleNamespace(model_name='LoraKGE_Layers',using_various_ranks=True,using_various_ranks_reverse=False,device=torch.device('cpu'),snapshot=0,snapshot_test=0,valid=False,predict_result=False,test_FWT=False,observation_path=str(observation))
    kg=types.SimpleNamespace(snapshots={0:types.SimpleNamespace(hr2t_all={},num_ent=4)})
    processor=DevBatchProcessor.__new__(DevBatchProcessor);processor.args=args;processor.kg=kg;processor.data_loader=[(head,rel,tail,labels)]
    observed=processor.process_epoch(Model())
    args.observation_path=None;original_path=processor.process_epoch(Model())
    assert observed==original_path
    z=np.load(observation/'train0_test0_after.npz',allow_pickle=False)
    assert z['rank'].tolist()==[1,1,3]
    recomputed=round(float(np.mean(1/z['rank'])),4)
    assert observed['mrr']==recomputed
    z.close()
    instructor=main.Instructor.__new__(main.Instructor)
    args.snapshot=0;args.epoch=17;args.observation_path=str(observation);args.save_path=str(pathlib.Path(td)/'checkpoints');args.valid_metrics='mrr';pathlib.Path(args.save_path).mkdir()
    instructor.args=args;instructor.model=torch.nn.Linear(2,2);instructor.best_valid=.55
    instructor.save_model(is_best=True,lora=False)
    selected=json.loads((observation/'selected_checkpoint_snapshot0.json').read_text());assert selected['epoch']==17 and (observation/selected['checkpoint']).is_file()
    args.snapshot=1;args.epoch=18;instructor.save_model(is_best=True,lora=True)
    selected=json.loads((observation/'selected_checkpoint_snapshot1.json').read_text());assert selected['epoch']==18 and selected['lora_only'] and (observation/selected['checkpoint']).is_file()
    output={'kind':'SYNTHETIC_OBSERVATION_PATCH_CHECK_NOT_KGE_RESULTS','ranking_with_without_export_identical':True,'expected_ranks':[1,1,3],'native_metrics_match_export':True,'selected_checkpoint_full_and_lora_metadata_checked':True,'GPU_jobs_launched':0}
    (base/'observation_check.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
