#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
import pandas as pd, numpy as np

ROOT=Path('/data/lc/tzh')
HEAD=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()

def finalize_aux(exp_id, aux_weight):
    r=ROOT/'results'/exp_id
    P=pd.read_csv(r/'paired_component_deltas.csv')
    A=pd.read_csv(r/'training_audit.csv')
    O=pd.read_parquet(r/'oof_predictions.parquet')
    key=['sample_id','seed','variant','target']
    if O.duplicated(key).any(): raise RuntimeError(f'{exp_id}: duplicate OOF')
    if not np.isfinite(O[['y_true','pred']].to_numpy()).all(): raise RuntimeError(f'{exp_id}: nonfinite OOF')
    rows=P[P.variant.eq('C_within_aux')]
    strong=[]
    for t in sorted(rows.target.unique()):
        g=rows[rows.target.eq(t)].set_index('metric')
        if float(g.loc['overall','delta_ccc_mean_across_seeds'])>0 and int(g.loc['overall','positive_seeds'])>=2:
            strong.append(t)
    decision='pass' if len(strong)>=2 else ('mixed' if len(strong)>=1 else 'reject')
    audit={
      'experiment_id':exp_id,'validity':'valid','decision':decision,
      'task':'standard absolute IEMOCAP VAD','variant':'C_within_aux',
      'aux_weight':aux_weight,'wavlm_layer':12,'backbone_frozen':True,
      'outer_split':'5-session LOSO','seeds':[20261002,20261003,20261004],
      'test_speaker_true_mean_used_as_input':False,
      'positive_overall_targets':strong,
      'oof_duplicate_key_count':int(O.duplicated(key).sum()),
      'oof_nonfinite_count':int((~np.isfinite(O[['y_true','pred']].to_numpy())).sum()),
      'speaker_overlap_count':int(A.speaker_overlap.sum())
    }
    (r/'audit_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    (r/'run_metadata.json').write_text(json.dumps({
      'experiment_id':exp_id,'git_head':HEAD,'architecture':'frozen WavLM-L12 + shared MLP + direct y head + within auxiliary head',
      'aux_weight':aux_weight,'test_inference_requires':'current utterance WavLM embedding only'
    },indent=2)+'\n')
    print(exp_id,decision,strong)

def finalize_wonly():
    exp_id='EXP-20261003-05'; r=ROOT/'results'/exp_id
    O=pd.read_parquet(r/'oof_w_predictions.parquet'); A=pd.read_csv(r/'training_audit.csv')
    key=['sample_id','seed','target']
    if O.duplicated(key).any(): raise RuntimeError('wonly duplicate OOF')
    if not np.isfinite(O[['w_true','w_pred']].to_numpy()).all(): raise RuntimeError('wonly nonfinite')
    old=json.loads((r/'audit_summary.json').read_text())
    old.update({
      'validity':'valid','decision':'diagnostic',
      'oof_duplicate_key_count':int(O.duplicated(key).sum()),
      'oof_nonfinite_count':int((~np.isfinite(O[['w_true','w_pred']].to_numpy())).sum()),
      'speaker_overlap_count':int(A.speaker_overlap.sum()),
      'git_head':HEAD
    })
    (r/'audit_summary.json').write_text(json.dumps(old,indent=2)+'\n')
    (r/'run_metadata.json').write_text(json.dumps({
      'experiment_id':exp_id,'git_head':HEAD,'architecture':'frozen WavLM-L12 + MLP -> w-only VAD deviation',
      'evaluation_target':'w = standardized y - speaker center (diagnostic only)',
      'deployable_absolute_vad_inference':False
    },indent=2)+'\n')
    print(exp_id,'diagnostic complete')

finalize_aux('EXP-20261003-03',0.25)
finalize_aux('EXP-20261003-04',0.5)
finalize_wonly()
