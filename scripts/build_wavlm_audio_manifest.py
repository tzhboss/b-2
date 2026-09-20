#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd
import yaml

NEEDED = [
    "sample_id","dataset","source_parquet","source_row_index","speaker_id","emotion_raw",
    "total_duration_sec","gender_canonical","emotion_5class_candidate","task_eligible",
    "emotion_training_usable","f0_median_hz","pitch_relative_st","speech_lufs",
    "loudness_relative_lu","phoneme_articulation_rate","rate_relative_ratio"
]

def esd_path(row: pd.Series, standardized_root: Path) -> str:
    stem = Path(row["source_parquet"]).stem
    return str(
        standardized_root / "esd/data/embedded/en"
        / str(row["speaker_id"]) / str(row["emotion_raw"])
        / f"{stem}_row{int(row['source_row_index']):06d}.flac"
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    src=Path(os.environ[cfg["dataset"]["source_env"]]).resolve()
    locroot=Path(os.environ[cfg["dataset"]["locator_root_env"]]).resolve()
    standardized_root=Path(os.environ[cfg["dataset"]["standardized_audio_root_env"]]).resolve()
    outroot=Path(cfg["outputs"]["artifact_root"]); outroot.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(src,columns=NEEDED)
    df=df[df.dataset.isin(cfg["dataset"]["datasets"])].copy()
    path_map={}
    for d in ["mead_part0","ravdess_speech"]:
        p=locroot/f"dataset={d}"/"data_0.parquet"
        x=pd.read_parquet(p,columns=["sample_id","audio_path"])
        path_map.update(dict(zip(x.sample_id.astype(str),x.audio_path.astype(str))))
    paths=[]
    for _,r in df.iterrows():
        if r.dataset=="esd_english":
            paths.append(esd_path(r, standardized_root))
        else:
            paths.append(path_map.get(str(r.sample_id),""))
    df["audio_path"]=paths
    def path_exists(p):
        return isinstance(p, str) and p not in {"", "nan", "None"} and Path(p).is_file()
    df["audio_exists"]=df.audio_path.map(path_exists)
    bad=df.loc[~df.audio_exists].copy()
    if len(bad):
        bad.to_parquet(outroot/"audio_mapping_exclusions.parquet",index=False)
    df=df.loc[df.audio_exists].copy()
    if not bool(df.audio_exists.all()):
        raise RuntimeError("unexpected unresolved audio after exclusion")
    df.to_parquet(outroot/"audio_manifest.parquet",index=False)
    print(df.dataset.value_counts().to_dict())
    print("rows",len(df),"excluded_missing_audio",len(bad),"all_audio_exists",bool(df.audio_exists.all()))

if __name__=="__main__":
    main()
