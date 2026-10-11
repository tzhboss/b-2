#!/usr/bin/env python3
"""Explicit staging, native training, and native exposed evaluation. Never auto-selects a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def filtered_indices(data_dir, rows, stable=False):
    ep = data_dir / ("entity2textlong.txt" if (data_dir/"entity2textlong.txt").exists() else "entity2text.txt")
    entities = {s.split("\t")[0] for s in ep.read_text().splitlines()}
    relations = {s.split("\t")[0] for s in (data_dir/"relation2text.txt").read_text().splitlines()}
    return [i for i,r in enumerate(rows) if
            (r["triple"] if stable else r["cor"])[0] in entities and
            (r["triple"] if stable else r["cor"])[2] in entities and
            (r["triple"] if stable else r["cor"])[1] in relations]

def prepare(args, cfg):
    pin = json.loads((HERE/"author-source-sha256.json").read_text())
    for f, expected in pin.items():
        if sha(SOURCE/f) != expected:
            raise RuntimeError("Author source changed: " + f)
    spec = cfg["datasets"][args.dataset]
    root = args.baseline_root.resolve()
    data = root/"data/KGEditor-HF"/args.dataset
    model = root/"models/KGEditor-HF"/args.dataset
    for f in ["config.json", "vocab.txt", "added_tokens.json", "pytorch_model.bin"]:
        if not (model/f).is_file():
            raise RuntimeError("Model download incomplete: " + str(model/f))
    for f, expected in [("edit_train.jsonl",spec["train_count"]),("edit_test.jsonl",spec["test_count"]),("stable.jsonl",spec["stable_count"])]:
        actual = len(read_rows(data/f))
        if actual != expected:
            raise RuntimeError("Unexpected source count %s=%d; expected %d" % (f,actual,expected))
    vocab = (model/"vocab.txt").read_text().splitlines()
    if len(vocab) != 30522:
        raise RuntimeError("Expected original BERT base vocabulary")
    ep = data/("entity2textlong.txt" if (data/"entity2textlong.txt").exists() else "entity2text.txt")
    entities = [s.split("\t")[0] for s in ep.read_text().splitlines()]
    relations = (data/"relations.txt").read_text().splitlines()
    added = json.loads((model/"added_tokens.json").read_text())
    for kind, count, offset in [("ENTITY",len(entities),30522),("RELATION",len(relations),30522+len(entities))]:
        for i in range(count):
            if added.get("[%s_%d]" % (kind,i)) != offset+i:
                raise RuntimeError("HF token-index mismatch")
    if json.loads((model/"config.json").read_text())["vocab_size"] != 30522+len(entities)+len(relations):
        raise RuntimeError("Model vocabulary size mismatch")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    stage = run/"data_view"
    stage.mkdir(exist_ok=True)
    input_hashes = {}
    for path in data.iterdir():
        if path.is_file() and (path.suffix == ".txt" or path.name in ["stable.jsonl","edit_train.jsonl","edit_test.jsonl"]):
            target = stage/path.name
            if target.exists() and sha(target) != sha(path):
                raise RuntimeError("Existing staged input changed: " + str(target))
            if not target.exists():
                shutil.copyfile(str(path), str(target))
            input_hashes[path.name] = sha(path)
    split_map = {"train.jsonl":"edit_train.jsonl","dev.jsonl":"edit_test.jsonl","test.jsonl":"edit_test.jsonl"}
    for target, original in split_map.items():
        dest = stage/target
        if dest.exists() and sha(dest) != sha(data/original):
            raise RuntimeError("Existing split alias changed: "+str(dest))
        shutil.copyfile(str(data/original),str(dest))
    tok = run/"base_tokenizer"
    tok.mkdir(exist_ok=True)
    shutil.copyfile(str(model/"vocab.txt"),str(tok/"vocab.txt"))
    (tok/"tokenizer_config.json").write_text(json.dumps(dict(
        tokenizer_class="BertTokenizer",do_lower_case=True,model_max_length=512)))
    (tok/"special_tokens_map.json").write_text(json.dumps(dict(
        cls_token="[CLS]",sep_token="[SEP]",unk_token="[UNK]",pad_token="[PAD]",mask_token="[MASK]")))
    training_rows = read_rows(stage/"train.jsonl")
    validation_rows = read_rows(stage/"dev.jsonl")
    training_ids = filtered_indices(stage,training_rows)
    validation_ids = filtered_indices(stage,validation_rows)
    memory_ids = filtered_indices(stage,read_rows(stage/"stable.jsonl"),True)
    if len(memory_ids) != spec["stable_count"]:
        raise RuntimeError("Stable-row filtering changes row IDs; audit before running")
    # WN18RR checkpoint may not contain FFN extension parameters. Inspect load diagnostics at runtime.
    result = dict(dataset=args.dataset,source_commit=cfg["source_commit"],
        model_revision=cfg["model_revision"],dataset_revision=cfg["dataset_revision"],
        split_aliases=split_map,validation_exposure="edit_test used for checkpoint selection",
        input_sha256=input_hashes,model_file_bytes=(model/"pytorch_model.bin").stat().st_size,
        token_vocabulary_base=30522,entity_count=len(entities),relation_count=len(relations),
        filtered_train_row_ids=training_ids,filtered_validation_row_ids=validation_ids,
        edit_num=cfg["common_args"]["edit_num"],drop_last=True,
        train_dropped_count=len(training_ids)%cfg["common_args"]["edit_num"],
        eval_dropped_rows=validation_ids[(len(validation_ids)//cfg["common_args"]["edit_num"])*cfg["common_args"]["edit_num"]:],
        runtime_validation="not executed")
    (run/"source-preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    return run,stage,tok,model,result

def native_parser():
    sys.path.insert(0,str(SOURCE))
    import importlib.util
    spec = importlib.util.spec_from_file_location("author_launcher",SOURCE/"models/KGEditor/run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from pytorch_lightning import Trainer
    p = argparse.ArgumentParser()
    p.add_argument("--dirpath")
    p.add_argument("--save_top_k",type=int,default=10)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--data_dir")
    p.add_argument("--max_seq_length",type=int,default=64)
    p.add_argument("--task_name",default="edit")
    return Trainer.add_argparse_args(m.add_model_specific_args(p))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root",type=Path,required=True)
    p.add_argument("--dataset",choices=["E-FB15k237","E-WN18RR"],required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--config",type=Path,default=HERE/"config.json")
    p.add_argument("--mode",choices=["preflight","train-evaluate","evaluate"],default="preflight")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--evaluation-file",type=Path,help="Explicit independent JSONL; default is exposed edit_test")
    p.add_argument("--device",default="cuda:0",help="Evaluation device; main must assign CUDA_VISIBLE_DEVICES")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    run,stage,tok,base,preflight = prepare(args,cfg)
    if args.mode == "preflight":
        print(json.dumps({"status":"SOURCE_PREFLIGHT_PASS","report":str(run/"source-preflight.json"),
                          "runtime_checked":False}))
        return
    sys.path.insert(0,str(SOURCE))
    os.chdir(str(run)) # All native relative output writes stay in run-dir.
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from observations import ObservedKGEditor,ObservedCollator,summarize
    import src.models.models as author
    values = dict(cfg["common_args"])
    values.update({k:v for k,v in cfg["datasets"][args.dataset].items() if k in ["data_type","ex_size","kb_layer"]})
    values.update(ex_model_checkpoint=str(base),model_name_or_path=str(tok),
                  data_dir=str(stage),dirpath=str(run/"logger"))
    argv = [item for k,v in values.items() for item in ["--"+k,str(v)]]
    hparams = native_parser().parse_args(argv) # Strict: no silent ignored flags.
    pl.seed_everything(hparams.seed)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if args.mode == "train-evaluate":
        if (run/"training-complete.json").exists():
            raise RuntimeError("Completed run exists; use evaluate or a fresh run-dir")
        logger = TensorBoardLogger(hparams.dirpath,name=None)
        ckpt = ModelCheckpoint(monitor="Eval/hits3",mode="max",
            dirpath=str(Path(logger.log_dir)/"checkpoints"),save_top_k=hparams.save_top_k,
            filename="model-{epoch:02d}-{Eval/hits3:.4f}-{valid_flipped:4f}")
        trainer = pl.Trainer.from_argparse_args(hparams,logger=logger,
            callbacks=[ckpt,LearningRateMonitor(logging_interval="step")])
        # Original KGEditor class, fit loop, losses/optimizer/validation untouched.
        model = author.KGEditor(**vars(hparams))
        trainer.fit(model)
        if trainer.global_rank != 0:
            return
        checkpoint = Path(ckpt.best_model_path)
        if not checkpoint.is_file():
            raise RuntimeError("No best native checkpoint saved")
        (run/"training-complete.json").write_text(json.dumps(dict(
            best_model_path=str(checkpoint),best_model_score=float(ckpt.best_model_score),
            global_step=trainer.global_step,current_epoch=trainer.current_epoch,
            validation_exposure="native exposed evaluation"),indent=2))
        del model,trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif checkpoint is None:
        raise RuntimeError("evaluate requires --checkpoint")
    model = ObservedKGEditor.load_from_checkpoint(str(checkpoint),map_location="cpu")
    # Author logging suppresses some load warnings; serialized key audit is saved in provenance.
    model.eval()
    model.to(torch.device(args.device))
    label = "native exposed evaluation (checkpoint-selection edit_test)"
    records = preflight["filtered_validation_row_ids"]
    if args.evaluation_file:
        rows = read_rows(args.evaluation_file)
        used = read_rows(stage/"train.jsonl")+read_rows(stage/"dev.jsonl")
        def key(r):
            return (tuple(r["ori"]),tuple(r["cor"]),r["label"])
        def fact(r):
            return tuple(r["ori"])
        if set(map(key,rows)) & set(map(key,used)) or set(map(fact,rows)) & set(map(fact,used)):
            raise RuntimeError("Requested independent evaluation overlaps training/validation")
        dest = stage/"test.jsonl"
        shutil.copyfile(str(args.evaluation_file),str(dest))
        from src.data.processor import get_dataset
        from src.data.processor import KGProcessor
        from argparse import Namespace
        params = Namespace(**dict(model.hparams))
        processor = KGProcessor(model.tokenizer,params)
        model.data_val = get_dataset(params,processor,processor.get_labels(str(stage)),model.tokenizer,"test")
        records = filtered_indices(stage,rows)
        label = "external evaluation: exact edit/fact overlap rejected; independence provenance supplied by caller"
    loader = model.val_dataloader()
    loader.collate_fn = ObservedCollator(model.sampler,model.data_val,records,"external_test" if args.evaluation_file else "edit_test")
    export = run/("external-examples.jsonl" if args.evaluation_file else "native-exposed-examples.jsonl")
    if export.exists():
        raise RuntimeError("Export already exists; use a fresh run-dir to preserve previous evaluations")
    model.export_path = str(export)
    model.evaluation_label = label
    # Author validation has no accumulation into base model; each edit batch starts at frozen weights.
    with torch.no_grad():
        for batch_idx,batch in enumerate(loader):
            batch = {k:v.to(args.device) if torch.is_tensor(v) else v for k,v in batch.items()}
            model._eval(batch,batch_idx)
    summary = summarize(export)
    summary.update(evaluation_label=label,checkpoint=str(checkpoint),
                   checkpoint_sha256=sha(checkpoint),source_preflight=str(run/"source-preflight.json"),
                   config=values,export=str(export),
                   note="No test_step exists in author; manual loop invokes exact _eval. No validation_epoch_end invoked.")
    (run/("external-metrics.json" if args.evaluation_file else "native-exposed-metrics.json")).write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
