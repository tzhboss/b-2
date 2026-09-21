from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path('/data/lc/tzh')
OUT=ROOT/'paper_assets/reference_frame_matching_20260921'
OUT.mkdir(parents=True,exist_ok=True)

c=pd.read_csv(ROOT/'results/EXP-20260921-12/effect_curves.csv')
c=c[(c.model_family=='linear_ridge') & (c.target.isin(['arousal','dominance']))].copy()

fig,ax=plt.subplots(figsize=(8.2,5.4))
for (ds,t),g in c.groupby(['dataset','target']):
    g=g.sort_values('lambda')
    ax.plot(g['lambda'],g['relative_minus_absolute_mean'],marker='o',label=f"{ds.upper()} {t.capitalize()}")
ax.axhline(0,linewidth=1)
ax.set_xlabel('Between-speaker target strength λ')
ax.set_ylabel('CCC(Relative) − CCC(Absolute)')
ax.set_title('Target-reference matching with semantically matched Pitch + Rate')
ax.legend(frameon=False,ncol=2)
fig.tight_layout()
fig.savefig(OUT/'fig1_target_reference_matching_pitch_rate.png',dpi=220)
plt.close(fig)

s=pd.read_csv(ROOT/'results/EXP-20260920-21/summary.csv')
s=s[(s.representation.isin(['kshot_hybrid','marginal_oracle_hybrid'])) &
    (s.target.isin(['arousal','dominance']))].copy()
fig,ax=plt.subplots(figsize=(8.2,5.4))
for t in ['arousal','dominance']:
    g=s[(s.target==t)&(s.representation=='kshot_hybrid')].sort_values('K')
    ax.plot(g.K,g.ccc_mean,marker='o',label=f'K-shot Hybrid — {t.capitalize()}')
    og=s[(s.target==t)&(s.representation=='marginal_oracle_hybrid')]
    oracle=float(og.ccc_mean.mean())
    ax.axhline(oracle,linestyle='--',linewidth=1.2,label=f'Marginal oracle — {t.capitalize()}')
ax.set_xlabel('Unlabeled enrollment utterances K')
ax.set_ylabel('Speaker-balanced CCC')
ax.set_title('Deployment-style K-shot speaker reference on MSP')
ax.legend(frameon=False,ncol=2)
fig.tight_layout()
fig.savefig(OUT/'fig2_kshot_reference_saturation_msp.png',dpi=220)
plt.close(fig)

d=pd.read_csv(ROOT/'results/EXP-20260920-02/paired_deltas.csv')
d=d[(d.comparison=='relative-absolute') &
    (d.attribute_set=='pitch') &
    (d.task.isin(['emotion','gender']))].copy()
order=['esd_english','mead_part0','ravdess_speech']
labels={'esd_english':'ESD','mead_part0':'MEAD','ravdess_speech':'RAVDESS'}
x=np.arange(len(order)); width=0.36
fig,ax=plt.subplots(figsize=(8.2,5.4))
emo=[float(d[(d.dataset==ds)&(d.task=='emotion')].delta_macro_f1_mean.iloc[0]) for ds in order]
gen=[float(d[(d.dataset==ds)&(d.task=='gender')].delta_macro_f1_mean.iloc[0]) for ds in order]
ax.bar(x-width/2,emo,width,label='Emotion')
ax.bar(x+width/2,gen,width,label='Gender')
ax.axhline(0,linewidth=1)
ax.set_xticks(x,[labels[z] for z in order])
ax.set_ylabel('Macro-F1(Relative) − Macro-F1(Absolute)')
ax.set_title('Pitch reference frame redistributes state vs trait information')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/'fig3_emotion_gender_pitch_reversal.png',dpi=220)
plt.close(fig)

w=pd.read_csv(ROOT/'results/EXP-20260920-04/pitch_decodability_summary.csv')
fig,ax=plt.subplots(figsize=(8.2,5.4))
for target,label in [
    ('absolute_pitch_semitone','Absolute pitch'),
    ('relative_pitch_semitone','Relative pitch'),
    ('implied_speaker_baseline_semitone','Implied speaker baseline')]:
    g=w[(w.dataset=='esd_english')&(w.target==target)].sort_values('layer')
    ax.plot(g.layer,g.r2_mean,label=label)
ax.set_xlabel('WavLM hidden layer')
ax.set_ylabel('Cross-validated R²')
ax.set_title('WavLM retains multiple prosodic reference frames (ESD)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/'fig4_wavlm_reference_decodability_esd.png',dpi=220)
plt.close(fig)

sl=pd.read_csv(ROOT/'results/EXP-20260921-12/slope_tests.csv')
sl[(sl.effect=='relative_minus_absolute') & (sl.target.isin(['arousal','dominance']))].to_csv(
    OUT/'table1_confirmatory_pitch_rate_slopes.csv',index=False)

c[c['lambda']==0][['dataset','model_family','target','relative_minus_absolute_mean']].to_csv(
    OUT/'table2_within_speaker_endpoints.csv',index=False)

d[['dataset','task','delta_macro_f1_mean','ci95_low','ci95_high']].to_csv(
    OUT/'table3_categorical_pitch_reversal.csv',index=False)

s[s.representation=='kshot_hybrid'][['K','target','ccc_mean','ccc_std']].to_csv(
    OUT/'table4_kshot_msp.csv',index=False)

sp=pd.read_csv(ROOT/'results/EXP-20260921-11/summary.csv')
sp=sp[sp.attribute_set=='all'][['dataset','representation','macro_f1_mean','macro_f1_std']]
sp.to_csv(OUT/'table5_speaker_identity_decodability.csv',index=False)

rows=[
{
'claim':'Speaker-relative prosody can improve affective-state prediction while suppressing stable speaker-trait information.',
'main_evidence':'EXP-20260920-02',
'status':'Supported with scope',
'key_result':'Pitch Relative−Absolute: emotion +0.059 to +0.091 Macro-F1 across ESD/MEAD/RAVDESS; gender −0.133 to −0.379.',
'paper_role':'Main motivation / empirical reversal',
'limitation':'Gender is a strong trait example; speaker identity suppression is only modest (EXP-20260921-11).'
},
{
'claim':'Reference-frame preference depends on the target reference frame, not on a universally superior normalization.',
'main_evidence':'EXP-20260921-12',
'status':'Confirmatory pass',
'key_result':'Pitch+Rate Relative−Absolute slope vs λ is significantly negative for Arousal/Dominance in MSP and IEMOCAP under linear and quadratic Ridge.',
'paper_role':'Primary mechanism result',
'limitation':'Continuous VAD external validation uses only MSP and IEMOCAP.'
},
{
'claim':'Pure within-speaker affect targets consistently prefer Relative prosody.',
'main_evidence':'EXP-20260921-12 / EXP-20260920-17',
'status':'Confirmatory pass',
'key_result':'At λ=0, all 8 corpus×target×model Arousal/Dominance cells have positive Relative−Absolute CCC.',
'paper_role':'Primary mechanism endpoint',
'limitation':'Valence remains weak and does not show the same strong pattern.'
},
{
'claim':'Stable speaker centers matter only when aligned to the correct speaker and when the target contains speaker-level structure.',
'main_evidence':'EXP-20260921-06',
'status':'Pass',
'key_result':'MSP raw true-center vs permuted-center: +0.296 Arousal, +0.236 Dominance CCC; residual effects ≈+0.001.',
'paper_role':'Mechanism intervention',
'limitation':'IEMOCAP raw effects are much smaller because between-speaker VAD structure is weak.'
},
{
'claim':'A small unlabeled enrollment set can recover useful speaker-reference information.',
'main_evidence':'EXP-20260920-21',
'status':'Supported / mixed strict shape',
'key_result':'K=20 Hybrid is within 0.0043 CCC of marginal oracle for Arousal and essentially identical for Dominance on a fixed downstream pool.',
'paper_role':'Deployment section',
'limitation':'Strict diminishing-return shape was not supported; K selection is corpus-dependent.'
},
{
'claim':'The target-reference mechanism is not an artifact of one model family.',
'main_evidence':'EXP-20260921-04 / EXP-20260921-12',
'status':'Pass',
'key_result':'Negative Arousal/Dominance slopes persist under both linear and quadratic Ridge in both corpora.',
'paper_role':'Robustness',
'limitation':'Earlier class-level categorical effects are more classifier-sensitive (EXP-20260920-15).'
},
{
'claim':'WavLM preserves information corresponding to multiple prosodic reference frames.',
'main_evidence':'EXP-20260920-04',
'status':'Supported for decodability',
'key_result':'Absolute, Relative, and implied speaker-baseline pitch are all highly decodable, with reference-frame-specific layer profiles.',
'paper_role':'Representation analysis / appendix or secondary main result',
'limitation':'Explicit Relative-vs-Absolute emotion gain at middle layer was not supported.'
},
{
'claim':'IEMOCAP provides a clean loudness replication.',
'main_evidence':'EXP-20260921-09',
'status':'Do not claim',
'key_result':'Replacing undocumented relative_db with RMS dB removed/reversed the loudness mechanism.',
'paper_role':'Sensitivity / limitation',
'limitation':'IEMOCAP loudness must be excluded from main evidence.'
},
]
pd.DataFrame(rows).to_csv(OUT/'claim_evidence_matrix.csv',index=False)

with open(OUT/'PAPER_EVIDENCE_SYNTHESIS.md','w') as f:
    f.write('# Prosodic Reference Frames — Paper Evidence Synthesis\n\n')
    f.write('## Core thesis\n\n')
    f.write('Prosodic normalization is an information transformation rather than a universally beneficial preprocessing step. ')
    f.write('The preferred acoustic reference frame depends on the reference frame of the prediction target.\n\n')
    f.write('A compact decomposition is x_su = μ_s + δ_su and y_su = speaker_mean_s + ε_su.\n\n')
    f.write('- Absolute prosody retains stable speaker baseline plus utterance deviation.\n')
    f.write('- Relative prosody emphasizes within-speaker deviation.\n')
    f.write('- Hybrid exposes both components.\n')
    f.write('- A target dominated by within-speaker state should prefer Relative.\n')
    f.write('- A target containing stable between-speaker structure can benefit from Absolute/Hybrid.\n\n')
    f.write('## Main-paper experiment stack\n\n')
    f.write('1. Categorical state/trait reversal — EXP-20260920-02.\n')
    f.write('2. Target decomposition — EXP-20260920-17.\n')
    f.write('3. Semantically matched cross-corpus intervention — EXP-20260921-12.\n')
    f.write('4. Speaker-center permutation — EXP-20260921-06.\n')
    f.write('5. K-shot deployment — EXP-20260920-21.\n')
    f.write('6. WavLM representation analysis — EXP-20260920-04.\n\n')
    f.write('## Claims that should be constrained\n\n')
    f.write('- Do not claim Relative normalization universally improves emotion prediction.\n')
    f.write('- Do not claim speaker identity is removed by Relative prosody; EXP-20260921-11 shows only modest suppression.\n')
    f.write('- Do not use IEMOCAP relative_db as primary loudness evidence; EXP-20260921-09 rejected semantic robustness.\n')
    f.write('- Do not claim monotonic K-shot mechanism recovery on IEMOCAP; EXP-20260921-08 found non-monotonic slope recovery.\n')
    f.write('- Treat Valence as a weak/negative-control dimension for the low-dimensional prosody story.\n\n')
    f.write('## Recommended main figures\n\n')
    f.write('- Fig. 1: fig1_target_reference_matching_pitch_rate.png — primary mechanism curve.\n')
    f.write('- Fig. 2: fig2_kshot_reference_saturation_msp.png — deployment practicality.\n')
    f.write('- Fig. 3: fig3_emotion_gender_pitch_reversal.png — intuitive state-vs-trait motivation.\n')
    f.write('- Fig. 4: fig4_wavlm_reference_decodability_esd.png — representation analysis, optional main/appendix depending space.\n\n')
    f.write('## Recommended paper framing\n\n')
    f.write('Working claim: Representation reference frame should match target reference frame.\n\n')
    f.write('This is stronger and more defensible than saying speaker-relative prosody is always better, because it explains both wins and failures under one mechanism.\n')

print('wrote',OUT)
for p in sorted(OUT.iterdir()):
    print(p.name,p.stat().st_size)
