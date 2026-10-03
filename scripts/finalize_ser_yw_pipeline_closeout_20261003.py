#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, subprocess
import pandas as pd
import yaml

ROOT=Path("/data/lc/tzh")

def decision_from_raw(path):
    s=pd.read_csv(path)
    ok=s[(s.delta_raw_ccc_mean>0)&(s.positive_seeds.ge(2))]
    n=len(ok)
    return ("pass" if n>=2 else ("mixed" if n==1 else "reject")), ok.target.astype(str).tolist(), s

def write_exp(exp_id,title,protocol,parents,status,extra):
    p=ROOT/"experiments"/exp_id/"experiment.yaml"; p.parent.mkdir(parents=True,exist_ok=True)
    obj={
      "schema_version":1,"experiment_id":exp_id,"title":title,
      "research_question":extra.pop("research_question"),
      "registered_hypothesis":extra.pop("registered_hypothesis"),
      "status":status,
      "parent_experiments":parents,
      "protocol_reference":protocol,
      **extra
    }
    p.write_text(yaml.safe_dump(obj,sort_keys=False,allow_unicode=True))
    return p

def append_registry(lines):
    p=ROOT/"EXPERIMENTS.md"; text=p.read_text()
    new=[]
    for exp_id,line in lines:
        if exp_id not in text:
            new.append(line)
    if new:
        with p.open("a") as f:
            f.write("\n"+"\n".join(new)+"\n")

def main():
    required=[
      ROOT/"results/EXP-20261003-03/audit_summary.json",
      ROOT/"results/EXP-20261003-04/audit_summary.json",
      ROOT/"results/EXP-20261003-05/audit_summary.json",
      ROOT/"results/EXP-20261003-06/standard_raw_ccc_summary.csv",
      ROOT/"results/EXP-20261003-07/standard_raw_ccc_summary.csv"
    ]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError("closeout missing required results: "+str(missing))

    d06,strong06,s06=decision_from_raw(ROOT/"results/EXP-20261003-06/standard_raw_ccc_summary.csv")
    d07,strong07,s07=decision_from_raw(ROOT/"results/EXP-20261003-07/standard_raw_ccc_summary.csv")

    e03=write_exp(
      "EXP-20261003-03","WavLM-L12 within-auxiliary lambda_w=0.25 ablation",
      "configs/experiments/EXP-20261003-03.yaml",["EXP-20261002-07"],
      {"execution":"completed","validity":"valid","decision":"pass"},
      {"research_question":"Does a weaker within-speaker auxiliary weight retain standard absolute VAD gains?",
       "registered_hypothesis":"lambda_w=0.25 is a sensitivity point; no target-specific superiority over lambda_w=1 was preregistered.",
       "runtime":{"seeds":[20261002,20261003,20261004],"outer_split":"5-session LOSO","wavlm_layer":12,"backbone_frozen":True},
       "role":"lambda sensitivity for y+w"}
    )
    e04=write_exp(
      "EXP-20261003-04","WavLM-L12 within-auxiliary lambda_w=0.50 ablation",
      "configs/experiments/EXP-20261003-04.yaml",["EXP-20261002-07"],
      {"execution":"completed","validity":"valid","decision":"pass"},
      {"research_question":"Does an intermediate within-speaker auxiliary weight retain standard absolute VAD gains?",
       "registered_hypothesis":"lambda_w=0.50 is a sensitivity point; no target-specific superiority over lambda_w=1 was preregistered.",
       "runtime":{"seeds":[20261002,20261003,20261004],"outer_split":"5-session LOSO","wavlm_layer":12,"backbone_frozen":True},
       "role":"lambda sensitivity for y+w"}
    )
    e05=write_exp(
      "EXP-20261003-05","WavLM-L12 w-only speaker-centered continuous VAD diagnostic",
      "configs/experiments/EXP-20261003-05.yaml",["EXP-20261002-07","EXP-20260921-16"],
      {"execution":"completed","validity":"valid","decision":"diagnostic"},
      {"research_question":"How predictable is speaker-centered continuous VAD w from the same frozen WavLM representation?",
       "registered_hypothesis":"w-only is diagnostic and is evaluated against w itself, not as an absolute-y predictor.",
       "runtime":{"seeds":[20261002,20261003,20261004],"outer_split":"5-session LOSO","wavlm_layer":12,"backbone_frozen":True},
       "role":"motivation diagnostic for relative auxiliary supervision"}
    )
    e06=write_exp(
      "EXP-20261003-06","MSP-Podcast frozen WavLM y versus y+w replication",
      "configs/experiments/EXP-20261003-06.yaml",["EXP-20261002-07","EXP-20260920-16"],
      {"execution":"completed","validity":"valid","decision":d06},
      {"research_question":"Does training-time speaker-centered VAD auxiliary supervision improve standard absolute MSP-Podcast VAD prediction under speaker-disjoint evaluation?",
       "registered_hypothesis":"PASS if at least two of V/A/D have positive mean standard raw CCC delta for y+w vs y-only with at least 2/3 positive seeds; MIXED if exactly one; otherwise REJECT.",
       "runtime":{"seeds":[20260920,20260921,20260922],"outer_split":"5 speaker-disjoint folds","wavlm_layer":12,"backbone_frozen":True},
       "primary_positive_targets":strong06,
       "role":"cross-dataset replication of y+w"}
    )
    effective_cfg="configs/experiments/EXP-20261003-07-runtime.yaml" if (ROOT/"configs/experiments/EXP-20261003-07-runtime.yaml").is_file() else "configs/experiments/EXP-20261003-07.yaml"
    e07=write_exp(
      "EXP-20261003-07","IEMOCAP fine-tuned WavLM y versus y+w",
      effective_cfg,["EXP-20261002-07"],
      {"execution":"completed","validity":"valid","decision":d07},
      {"research_question":"Does speaker-centered VAD auxiliary supervision improve standard absolute VAD when the WavLM encoder itself is fine-tuned?",
       "registered_hypothesis":"PASS if at least two of V/A/D have positive mean standard raw CCC delta for y+w vs y-only with at least 2/3 positive seeds; MIXED if exactly one; otherwise REJECT.",
       "runtime":{"seeds":[20261002,20261003,20261004],"outer_split":"5-session LOSO","feature_extractor_cnn_frozen":True,"transformer_encoder_trainable":True},
       "primary_positive_targets":strong07,
       "role":"fine-tuned-backbone validation of y+w"}
    )

    append_registry([
      ("EXP-20261003-03", "| EXP-20261003-03 | WavLM-L12 within-auxiliary lambda_w=0.25 ablation | completed | valid | pass | iemocap_wavlm_within_aux_weight_v1 | EXP-20261002-07 |"),
      ("EXP-20261003-04", "| EXP-20261003-04 | WavLM-L12 within-auxiliary lambda_w=0.50 ablation | completed | valid | pass | iemocap_wavlm_within_aux_weight_v1 | EXP-20261002-07 |"),
      ("EXP-20261003-05", "| EXP-20261003-05 | WavLM-L12 w-only speaker-centered continuous VAD diagnostic | completed | valid | diagnostic | iemocap_wavlm_wonly_v1 | EXP-20261002-07, EXP-20260921-16 |"),
      ("EXP-20261003-06", f"| EXP-20261003-06 | MSP-Podcast frozen WavLM y versus y+w replication | completed | valid | {d06} | msp_wavlm_within_aux_v1 | EXP-20261002-07, EXP-20260920-16 |"),
      ("EXP-20261003-07", f"| EXP-20261003-07 | IEMOCAP fine-tuned WavLM y versus y+w | completed | valid | {d07} | iemocap_wavlm_e2e_within_aux_v1 | EXP-20261002-07 |")
    ])

    lam=[]
    for val,exp in [(0.25,"EXP-20261003-03"),(0.5,"EXP-20261003-04"),(1.0,"EXP-20261002-07")]:
        p=pd.read_csv(ROOT/"results"/exp/"paired_component_deltas.csv")
        p=p[(p.variant=="C_within_aux")&(p.metric=="overall")]
        for _,r in p.iterrows():
            lam.append({"lambda_w":val,"target":r.target,"delta_overall_ccc":r.delta_ccc_mean_across_seeds,
                        "ci95_low":r.ci95_low,"ci95_high":r.ci95_high,"positive_seeds":int(r.positive_seeds)})
    for t in ["arousal","valence","dominance"]:
        lam.append({"lambda_w":0.0,"target":t,"delta_overall_ccc":0.0,"ci95_low":0.0,"ci95_high":0.0,"positive_seeds":0})
    L=pd.DataFrame(lam).sort_values(["target","lambda_w"])
    w=pd.read_csv(ROOT/"results/EXP-20261003-05/w_metrics_summary.csv")
    p06=pd.read_csv(ROOT/"results/EXP-20261003-06/paired_component_deltas.csv")
    p07=pd.read_csv(ROOT/"results/EXP-20261003-07/paired_component_deltas.csv")
    report=ROOT/"reports/ser_yw_pipeline_20261003.md"; report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(
      "# SER y+w validation pipeline — 2026-10-03\n\n"
      "## Main formulation\n\n"
      "Standard absolute VAD remains the inference target. The method adds training-only speaker-centered VAD supervision "
      "w = y - speaker mean and predicts y only at test time.\n\n"
      "## Lambda sensitivity (frozen WavLM-L12, IEMOCAP)\n\n"
      +L.to_markdown(index=False)+"\n\n"
      "## w-only diagnostic\n\n"+w.to_markdown(index=False)+"\n\n"
      "## MSP-Podcast frozen-WavLM replication\n\n"
      +s06.to_markdown(index=False)+"\n\nDecision: **"+d06+"**; positive primary targets: "+", ".join(strong06)+".\n\n"
      "Component deltas:\n\n"+p06.to_markdown(index=False)+"\n\n"
      "## IEMOCAP fine-tuned WavLM replication\n\n"
      +s07.to_markdown(index=False)+"\n\nDecision: **"+d07+"**; positive primary targets: "+", ".join(strong07)+".\n\n"
      "Component deltas:\n\n"+p07.to_markdown(index=False)+"\n"
    )

    paths=[
      "EXPERIMENTS.md",
      "configs/experiments/EXP-20261003-03.yaml",
      "configs/experiments/EXP-20261003-04.yaml",
      "configs/experiments/EXP-20261003-05.yaml",
      "configs/experiments/EXP-20261003-06.yaml",
      "configs/experiments/EXP-20261003-07.yaml",
      "scripts/run_iemocap_wavlm_wonly_20261003.py",
      "scripts/finalize_yw_aux_followups_20261003.py",
      "scripts/build_msp_wavlm_manifest_20261003.py",
      "scripts/extract_msp_wavlm_layers_20261003.py",
      "scripts/supervise_msp_wavlm_extract_remaining_20261003.py",
      "scripts/run_msp_wavlm_within_aux_shard_20261003.py",
      "scripts/merge_msp_wavlm_within_aux_20261003.py",
      "scripts/run_iemocap_wavlm_e2e_within_aux_shard_20261003.py",
      "scripts/smoke_iemocap_wavlm_e2e_20261003.py",
      "scripts/merge_iemocap_wavlm_e2e_within_aux_20261003.py",
      "scripts/supervise_ser_yw_pipeline_20261003.py",
      "scripts/finalize_ser_yw_pipeline_closeout_20261003.py",
      "experiments/EXP-20261003-03/experiment.yaml",
      "experiments/EXP-20261003-04/experiment.yaml",
      "experiments/EXP-20261003-05/experiment.yaml",
      "experiments/EXP-20261003-06/experiment.yaml",
      "experiments/EXP-20261003-07/experiment.yaml",
      "reports/ser_yw_pipeline_20261003.md"
    ]
    runtime=ROOT/"configs/experiments/EXP-20261003-07-runtime.yaml"
    if runtime.is_file(): paths.append("configs/experiments/EXP-20261003-07-runtime.yaml")
    subprocess.run(["git","-C",str(ROOT),"add","--",*paths],check=True)
    diff=subprocess.run(["git","-C",str(ROOT),"diff","--cached","--quiet"]).returncode
    if diff!=0:
        subprocess.run(["git","-C",str(ROOT),"commit","-m","Finalize y+w SER validation pipeline"],check=True)
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    print(json.dumps({"EXP-20261003-06":d06,"EXP-20261003-07":d07,"head":head},indent=2))

if __name__=="__main__":
    main()
