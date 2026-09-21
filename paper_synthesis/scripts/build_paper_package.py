#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path('/data/lc/tzh')
OUT=ROOT/'paper_synthesis'
TAB=OUT/'tables'
FIG=OUT/'figures'
TAB.mkdir(parents=True,exist_ok=True)
FIG.mkdir(parents=True,exist_ok=True)

d02=pd.read_csv(ROOT/'results/EXP-20260920-02/paired_deltas.csv')
t1=d02[
    (d02.comparison=='relative-absolute') &
    (d02.attribute_set.isin(['pitch','all'])) &
    (d02.task.isin(['emotion','gender']))
].copy()
t1=t1.rename(columns={'delta_macro_f1_mean':'delta_relative_minus_absolute','dataset':'corpus','attribute_set':'attribute'})
t1[['corpus','task','attribute','delta_relative_minus_absolute','ci95_low','ci95_high']].to_csv(
    TAB/'table1_explicit_prosody_task_reversal.csv',index=False)

v17=pd.read_csv(ROOT/'results/EXP-20260920-17/variance_decomposition.csv')
v17=v17[v17.attribute_set=='all'][['target','icc_like']].drop_duplicates()
v17['corpus']='MSP'
v02=pd.read_csv(ROOT/'results/EXP-20260921-02/variance_decomposition.csv')
v02=v02[['target','icc_like']].copy(); v02['corpus']='IEMOCAP'
icc=pd.concat([v17,v02],ignore_index=True)
sl03=pd.read_csv(ROOT/'results/EXP-20260921-03/slope_tests.csv')
sl03=sl03[sl03.effect=='relative_minus_absolute'][['dataset','target','slope_mean','ci95_low','ci95_high']]
sl03['corpus']=sl03.dataset.str.upper()
sl03=sl03.drop(columns='dataset')
t2=icc.merge(sl03,on=['corpus','target'],how='left')
t2.to_csv(TAB/'table2_target_structure_and_coupling.csv',index=False)

sl04=pd.read_csv(ROOT/'results/EXP-20260921-04/slope_tests.csv')
t3=sl04[(sl04.effect=='relative_minus_absolute')&(sl04.target.isin(['arousal','dominance']))].copy()
t3.to_csv(TAB/'table3_model_robustness_slopes.csv',index=False)

sl05=pd.read_csv(ROOT/'results/EXP-20260921-05/slope_tests.csv')
t4=sl05[(sl05.effect=='relative_minus_absolute')&(sl05.target.isin(['arousal','dominance']))].copy()
t4.to_csv(TAB/'table4_attribute_specific_slopes.csv',index=False)

p06=pd.read_csv(ROOT/'results/EXP-20260921-06/effect_summary.csv')
p06.to_csv(TAB/'table5_speaker_center_permutation.csv',index=False)

p19=pd.read_csv(ROOT/'results/EXP-20260920-19/paired_deltas.csv')
p19=p19[
    (p19.K==10)&
    (p19.target.isin(['arousal','dominance']))&
    (p19.comparison.isin(['kshot_hybrid-absolute','kshot_hybrid-marginal_oracle_hybrid']))
].copy()
sat=pd.read_csv(ROOT/'results/EXP-20260921-01/summary.csv')
sat=sat[
    (sat.target.isin(['arousal','dominance']))&
    (sat.representation=='kshot_hybrid')&
    (sat.K.isin([10,20,50]))
].copy()
p19.to_csv(TAB/'table6a_kshot_k10_deltas.csv',index=False)
sat.to_csv(TAB/'table6b_kshot_saturation.csv',index=False)

safe=pd.read_csv(ROOT/'results/EXP-20260921-10/slope_tests.csv')
safe=safe[(safe.target.isin(['arousal','dominance']))&(safe.effect=='relative_minus_absolute')].copy()
safe.to_csv(TAB/'table7_iemocap_semantically_safe_slopes.csv',index=False)

claims=[
("C1","Explicit prosody reference frame is task-dependent","EXP-20260920-02","supported",
 "Relative pitch improves categorical emotion but strongly hurts gender/trait decoding across ESD, MEAD, RAVDESS."),
("C2","Categorical emotion reference preference is class- and attribute-dependent","EXP-20260920-09; EXP-20260920-10","supported",
 "Per-emotion and per-attribute effects change sign; no universal Relative winner."),
("C3","WavLM retains both absolute and relative reference information","EXP-20260920-04","supported",
 "Absolute, relative, and implied baseline pitch are highly decodable; relative information weakens in later layers."),
("C4","Raw VAD preference depends on target between-speaker structure","EXP-20260920-17; EXP-20260921-02; EXP-20260921-03","supported",
 "Controlled lambda intervention yields significantly negative Relative-minus-Absolute slopes for Arousal/Dominance in MSP and IEMOCAP."),
("C5","Target-reference coupling is model-family robust","EXP-20260921-04","supported",
 "Linear and quadratic Ridge preserve negative Arousal/Dominance slopes in both corpora."),
("C6","Coupling is attribute-dependent, not pitch-only","EXP-20260921-05","supported",
 "Pitch, loudness, and rate all show coupling somewhere; magnitude differs strongly by corpus/attribute."),
("C7","Correct speaker-center identity causes raw-VAD Hybrid utility","EXP-20260921-06","supported",
 "Permuting speaker centers destroys large MSP raw Arousal/Dominance gains while residual targets are nearly unchanged."),
("C8","Unlabeled enrollment can recover useful speaker reference","EXP-20260920-19; EXP-20260921-01; EXP-20260921-07; EXP-20260921-08","supported_with_boundary",
 "K-shot Hybrid is useful; K about 20 is near practical saturation in MSP. IEMOCAP K-shot slope recovery is noisier/non-monotonic."),
("C9","IEMOCAP external replication survives without ambiguous loudness","EXP-20260921-10","supported",
 "Pitch+rate slopes remain significantly negative for Arousal/Dominance under linear and quadratic Ridge.")
]
pd.DataFrame(claims,columns=['claim_id','claim','evidence','status','boundary']).to_csv(TAB/'evidence_matrix.csv',index=False)

limits=[
("L1","EXP-20260920-01","invalid","Fold/class-universe bug; superseded by EXP-02."),
("L2","EXP-20260920-20","invalid","Runtime provenance mismatch; numerical outputs must not be used."),
("L3","EXP-20260921-09","reject","IEMOCAP relative_db is semantically ambiguous and not robust to RMS-dB replacement; do not use IEMOCAP loudness as main evidence."),
("L4","EXP-20260920-14/15","inconclusive","Some categorical class-level effects are classifier-family sensitive."),
("L5","EXP-20260920-03","inconclusive","Frozen final-layer WavLM does not reproduce a strong explicit Relative-vs-Absolute conditional reversal."),
("L6","Valence analyses","weak signal","Simple pitch/loudness/rate provide near-zero Valence prediction in MSP and weak effects in IEMOCAP.")
]
pd.DataFrame(limits,columns=['limit_id','source','status','implication']).to_csv(TAB/'limitations_and_exclusions.csv',index=False)

curves=pd.read_csv(ROOT/'results/EXP-20260921-03/effect_curves.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
for ax,target in zip(axes,['arousal','dominance']):
    sub=curves[curves.target==target]
    for ds,g in sub.groupby('dataset'):
        ax.plot(g['lambda'],g['relative_minus_absolute_mean'],marker='o',label=ds.upper())
    ax.axhline(0,linewidth=1)
    ax.set_title(target.capitalize())
    ax.set_xlabel('Between-speaker target strength lambda')
    ax.set_ylabel('CCC(Relative - Absolute)')
    ax.legend()
fig.tight_layout()
fig.savefig(FIG/'fig1_target_reference_intervention.png',dpi=220,bbox_inches='tight')
plt.close(fig)

mr=pd.read_csv(ROOT/'results/EXP-20260921-04/slope_tests.csv')
mr=mr[(mr.effect=='relative_minus_absolute')&(mr.target.isin(['arousal','dominance']))]
fig,ax=plt.subplots(figsize=(9,5))
labels=[]; vals=[]; lows=[]; highs=[]
for _,r in mr.iterrows():
    labels.append(f"{r['dataset'].upper()} {r['target'][0].upper()} {r['model_family'].replace('_ridge','')}")
    vals.append(r.slope_mean); lows.append(r.slope_mean-r.ci95_low); highs.append(r.ci95_high-r.slope_mean)
y=np.arange(len(vals))
ax.errorbar(vals,y,xerr=np.vstack([lows,highs]),fmt='o',capsize=3)
ax.axvline(0,linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel('Slope of CCC(Relative - Absolute) vs lambda')
ax.set_title('Target-reference coupling across corpus and model family')
fig.tight_layout()
fig.savefig(FIG/'fig2_model_robustness.png',dpi=220,bbox_inches='tight')
plt.close(fig)

perm=pd.read_csv(ROOT/'results/EXP-20260921-06/effect_summary.csv')
perm=perm[perm.target.isin(['arousal','dominance'])]
fig,ax=plt.subplots(figsize=(8,4.8))
labs=[]; vals=[]; lows=[]; highs=[]
for _,r in perm.iterrows():
    labs.append(f"{r['dataset'].upper()} {r['task']} {r['target'][0].upper()}")
    vals.append(r.delta_mean); lows.append(r.delta_mean-r.ci95_low); highs.append(r.ci95_high-r.delta_mean)
y=np.arange(len(vals))
ax.errorbar(vals,y,xerr=np.vstack([lows,highs]),fmt='o',capsize=3)
ax.axvline(0,linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(labs)
ax.set_xlabel('CCC(True center - Permuted center)')
ax.set_title('Correct speaker-center identity matters only when target contains speaker structure')
fig.tight_layout()
fig.savefig(FIG/'fig3_center_permutation.png',dpi=220,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.5))
for t,g in sat.groupby('target'):
    ax.plot(g.K,g.ccc_mean,marker='o',label=t.capitalize())
ax.set_xlabel('Unlabeled enrollment utterances K')
ax.set_ylabel('Speaker-balanced CCC')
ax.set_title('K-shot Hybrid raw-VAD sample efficiency (MSP)')
ax.legend()
fig.tight_layout()
fig.savefig(FIG/'fig4_kshot_saturation.png',dpi=220,bbox_inches='tight')
plt.close(fig)

text="""# Prosodic Reference Frames — Evidence Synthesis

## Central claim

The current experiments support a target-reference matching account rather than a universal
normalization advantage.

A speaker-level prosodic observation can be decomposed conceptually into speaker baseline,
within-speaker deviation, and other variation. A target can likewise contain a stable
between-speaker component plus a within-speaker component. The utility of Absolute, Relative, or
Hybrid prosody depends on which components the target rewards.

## Strongest evidence chain

1. Explicit controlled prosody: Relative pitch improves categorical emotion but removes
   speaker-trait information useful for gender classification (EXP-02).
2. Categorical heterogeneity: emotion-class and acoustic-attribute effects vary in sign
   (EXP-09/10); therefore Relative is better is not a valid universal claim.
3. Continuous VAD decomposition: MSP raw Arousal/Dominance contain large between-speaker
   components; after removing speaker target means, Relative becomes strongly better and baseline
   restoration becomes negligible (EXP-17).
4. External domain contrast: IEMOCAP has far less between-speaker VAD structure and correspondingly
   favors Relative much earlier (EXP-21-02).
5. Controlled intervention: continuously increasing between-speaker target strength makes
   Relative-minus-Absolute utility decrease in both corpora (EXP-21-03).
6. Model robustness: the intervention slope survives linear and quadratic Ridge (EXP-21-04).
7. Mechanism intervention: permuting speaker-center identity destroys raw MSP Hybrid utility but
   leaves within-speaker residual prediction nearly unchanged (EXP-21-06).
8. Deployment: unlabeled K-shot enrollment estimates useful speaker references; K around 20 is near
   a practical downstream plateau in MSP (EXP-20-19 / EXP-21-01 / EXP-21-07/08).
9. Semantically safe external replication: IEMOCAP pitch+rate alone preserve the coupling under
   both model families (EXP-21-10).

## Claims that should NOT be made

- Do not claim Relative prosody is universally superior.
- Do not claim Absolute prosody is universally superior for continuous affect.
- Do not use IEMOCAP relative_db as main loudness evidence; its semantics are unresolved and RMS-dB
  replacement reverses the relevant slope (EXP-21-09).
- Do not present the frozen-WavLM final-layer result as invariance; the representation retains highly
  decodable absolute, relative, and speaker-baseline pitch information.
- Do not use EXP-20 numerical results; that run is invalid due runtime provenance mismatch.
- Do not frame Valence as a strong success case for these low-dimensional prosodic features.

## Recommended paper-level statement

Prosodic normalization is an information transformation, not a universally beneficial nuisance
removal step. Speaker-relative representations are advantageous when the prediction target is
defined within speaker, whereas stable speaker baselines become useful when the target itself
contains between-speaker structure.

The controlled target intervention and center-identity permutation are the strongest mechanism
experiments supporting this statement.
"""
(OUT/'SYNTHESIS.md').write_text(text)

print('built',OUT)
for p in sorted(TAB.glob('*')): print(p.name,p.stat().st_size)
for p in sorted(FIG.glob('*')): print(p.name,p.stat().st_size)
