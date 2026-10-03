#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

SRC=Path('/data/lc/audio_feature_labeling/old/legacy_feature_pipeline/derived/final_dataset_enriched.parquet')
OUT=Path('/data/lc/tzh/artifacts/EXP-20261003-06/msp_manifest.parquet')
COLS=['sample_id','dataset','split','audio_path','speaker_id','arousal_mean_1_7','valence_mean_1_7','dominance_mean_1_7']

d=pd.read_parquet(SRC,columns=COLS)
d=d[d.dataset.eq('msp')].copy()
d=d.dropna(subset=['sample_id','audio_path','speaker_id','arousal_mean_1_7','valence_mean_1_7','dominance_mean_1_7'])
d=d[d.speaker_id.astype(str).ne('Unknown')].copy()
d['audio_path']=d.audio_path.astype(str)
d=d[d.audio_path.map(lambda p: Path(p).is_file())].copy()
d['speaker_id']=d.speaker_id.astype(str)
d=d.sort_values('sample_id').reset_index(drop=True)
OUT.parent.mkdir(parents=True,exist_ok=True)
d.to_parquet(OUT,index=False)
print('rows',len(d),'speakers',d.speaker_id.nunique())
print(d['split'].value_counts(dropna=False).to_string())
print('min_rows_per_speaker',int(d.groupby('speaker_id').size().min()))
print('max_rows_per_speaker',int(d.groupby('speaker_id').size().max()))
print('out',OUT)
