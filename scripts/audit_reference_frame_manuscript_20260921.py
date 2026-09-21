from pathlib import Path
import pandas as pd
import math

ROOT=Path('/data/lc/tzh')
checks=[]

def add(name,actual,expected,tol=5e-4):
    ok=abs(float(actual)-float(expected))<=tol
    checks.append((name,float(actual),float(expected),tol,ok))

# EXP02 categorical reversal
d=pd.read_csv(ROOT/'results/EXP-20260920-02/paired_deltas.csv')
for ds,emo,gen in [
    ('esd_english',0.090783,-0.132995),
    ('mead_part0',0.090361,-0.378994),
    ('ravdess_speech',0.058641,-0.229891),
]:
    q=d[(d.dataset==ds)&(d.attribute_set=='pitch')&(d.comparison=='relative-absolute')]
    add(f'{ds} emotion pitch delta',q[q.task=='emotion'].delta_macro_f1_mean.iloc[0],emo,1e-6)
    add(f'{ds} gender pitch delta',q[q.task=='gender'].delta_macro_f1_mean.iloc[0],gen,1e-6)

# EXP17 target decomposition
v=pd.read_csv(ROOT/'results/EXP-20260920-17/variance_decomposition.csv')
v=v[v.attribute_set=='all'].set_index('target')
for t,e in [('valence',0.213285),('arousal',0.417600),('dominance',0.346299)]:
    add(f'MSP ICC {t}',v.loc[t,'icc_like'],e,1e-6)

bs=pd.read_csv(ROOT/'results/EXP-20260920-17/between_speaker_summary.csv')
bs=bs[(bs.attribute_set=='all')&(bs.representation=='prosodic_baseline')].set_index('target')
for t,e in [('arousal',0.482495),('dominance',0.441162),('valence',0.022200)]:
    add(f'MSP baseline CCC {t}',bs.loc[t,'ccc_mean'],e,1e-6)

wd=pd.read_csv(ROOT/'results/EXP-20260920-17/within_speaker_deltas.csv')
wd=wd[(wd.attribute_set=='all')].set_index(['target','comparison'])
for t,e in [('arousal',0.130572),('dominance',0.092146),('valence',0.004947)]:
    add(f'MSP residual Relative-Absolute {t}',wd.loc[(t,'relative-absolute'),'delta_ccc_mean'],e,1e-6)

# EXP12 confirmatory slopes/endpoints
sl=pd.read_csv(ROOT/'results/EXP-20260921-12/slope_tests.csv')
exp_slopes={
('msp','linear_ridge','arousal'):-0.318119,
('msp','linear_ridge','dominance'):-0.182070,
('msp','quadratic_ridge','arousal'):-0.320660,
('msp','quadratic_ridge','dominance'):-0.184600,
('iemocap','linear_ridge','arousal'):-0.022447,
('iemocap','linear_ridge','dominance'):-0.010889,
('iemocap','quadratic_ridge','arousal'):-0.026461,
('iemocap','quadratic_ridge','dominance'):-0.017520,
}
for key,e in exp_slopes.items():
    ds,fam,t=key
    z=sl[(sl.dataset==ds)&(sl.model_family==fam)&(sl.target==t)&(sl.effect=='relative_minus_absolute')]
    add(f'EXP12 slope {ds} {fam} {t}',z.slope_mean.iloc[0],e,1e-6)

c=pd.read_csv(ROOT/'results/EXP-20260921-12/effect_curves.csv')
exp_ep={
('msp','linear_ridge','arousal'):0.138854,
('msp','linear_ridge','dominance'):0.072695,
('msp','quadratic_ridge','arousal'):0.138869,
('msp','quadratic_ridge','dominance'):0.074627,
('iemocap','linear_ridge','arousal'):0.136330,
('iemocap','linear_ridge','dominance'):0.045773,
('iemocap','quadratic_ridge','arousal'):0.155135,
('iemocap','quadratic_ridge','dominance'):0.048616,
}
for key,e in exp_ep.items():
    ds,fam,t=key
    z=c[(c.dataset==ds)&(c.model_family==fam)&(c.target==t)&(c['lambda']==0)]
    add(f'EXP12 endpoint {ds} {fam} {t}',z.relative_minus_absolute_mean.iloc[0],e,1e-6)

# EXP06 permutation
p=pd.read_csv(ROOT/'results/EXP-20260921-06/effect_summary.csv').set_index(['dataset','task','target'])
for key,e in [
(('msp','raw','arousal'),0.295501),
(('msp','raw','dominance'),0.236292),
(('msp','residual','arousal'),0.001393),
(('msp','residual','dominance'),0.001004),
]:
    add('EXP06 '+str(key),p.loc[key,'delta_mean'],e,1e-6)

# EXP21 Kshot
k=pd.read_csv(ROOT/'results/EXP-20260920-21/summary.csv')
for K,t,e in [
(1,'arousal',0.479489),(10,'arousal',0.487128),(20,'arousal',0.490285),(50,'arousal',0.494563),
(1,'dominance',0.379306),(10,'dominance',0.383675),(20,'dominance',0.383898),(50,'dominance',0.384030),
]:
    z=k[(k.K==K)&(k.target==t)&(k.representation=='kshot_hybrid')]
    add(f'EXP21 K{K} {t}',z.ccc_mean.iloc[0],e,1e-6)

# EXP04 WavLM maxima
w=pd.read_csv(ROOT/'results/EXP-20260920-04/pitch_decodability_summary.csv')
exp_w={
('esd_english','absolute_pitch_semitone'):(3,0.958254),
('esd_english','relative_pitch_semitone'):(3,0.894553),
('esd_english','implied_speaker_baseline_semitone'):(6,0.982723),
('mead_part0','absolute_pitch_semitone'):(3,0.987439),
('mead_part0','relative_pitch_semitone'):(4,0.860508),
('mead_part0','implied_speaker_baseline_semitone'):(4,0.970752),
('ravdess_speech','absolute_pitch_semitone'):(9,0.939498),
('ravdess_speech','relative_pitch_semitone'):(5,0.892427),
('ravdess_speech','implied_speaker_baseline_semitone'):(4,0.978946),
}
for key,(layer,r2) in exp_w.items():
    ds,t=key
    g=w[(w.dataset==ds)&(w.target==t)]
    z=g.loc[g.r2_mean.idxmax()]
    checks.append((f'EXP04 top layer {ds} {t}',float(z.layer),float(layer),0.0,int(z.layer)==layer))
    add(f'EXP04 top R2 {ds} {t}',z.r2_mean,r2,1e-6)

failed=[x for x in checks if not x[-1]]
out=ROOT/'paper_draft/MANUSCRIPT_NUMERIC_AUDIT.md'
with out.open('w') as f:
    f.write('# Manuscript Numeric Audit\n\n')
    f.write(f'- Checks: {len(checks)}\n')
    f.write(f'- Passed: {len(checks)-len(failed)}\n')
    f.write(f'- Failed: {len(failed)}\n\n')
    f.write('| Check | Actual | Expected | Tolerance | Pass |\n')
    f.write('|---|---:|---:|---:|---|\n')
    for name,a,e,tol,ok in checks:
        f.write(f'| {name} | {a:.6f} | {e:.6f} | {tol:.6g} | {"yes" if ok else "NO"} |\n')
if failed:
    print('FAILED')
    for x in failed: print(x)
    raise SystemExit(2)
print(f'PASS {len(checks)}/{len(checks)}')
