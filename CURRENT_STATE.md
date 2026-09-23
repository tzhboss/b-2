# CURRENT_STATE.md

## Phase

The reference-frame project now contains **41 completed registered experiments**: 38 valid and 3 invalid under audit. Current paper-level evidence supports the mechanism in interpretable prosody, frozen WavLM representations, and unlabeled enrollment.

Central claim:

> **Representation reference frame should match target reference frame.**

Relative representations emphasize within-speaker deviation. Stable speaker-center information becomes useful as the target retains more between-speaker structure.

## How the claim emerged

The project did not start with the final hypothesis.

1. **EXP-20260920-02**: corrected state-versus-trait pilot. Relative pitch improves emotion classification across ESD/MEAD/RAVDESS, while Absolute pitch preserves much more dataset-provided gender information.
2. **EXP-20260920-07**: natural-corpus counterexample. MSP aggregate categorical emotion prefers Absolute pitch, breaking any universal “Relative is better for emotion” claim.
3. **EXP-20260920-08/09**: the removed speaker baseline can itself be task-relevant; MSP’s categorical reversal is strongly class-dependent.
4. **EXP-20260920-16/17**: continuous MSP VAD reveals the key mechanism. Raw Arousal/Dominance contain substantial between-speaker structure; after target centering, preference reverses toward Relative.
5. **EXP-20260921-03/04**: direct target-reference intervention changes only target between-speaker structure while keeping inputs, speakers, folds, and model family fixed. This becomes the paper’s central mechanism test.

## Strongest current findings

### Target decomposition — EXP-20260920-17

MSP ICC-like between-speaker fractions:

- Valence ~0.213
- Arousal ~0.418
- Dominance ~0.346

Within-speaker residual target, Relative-minus-Absolute CCC:

- Arousal +0.1306
- Dominance +0.0921
- Valence +0.00495

Adding the stable baseline back after target centering has essentially zero effect.

### Correct speaker-level inference — EXP-20260921-15

Speaker-cluster bootstrap, 5,000 replicates.

MSP, 1,915 speakers:

- Linear A: -0.3188, 95% CI [-0.3384,-0.2998]
- Linear D: -0.1825, [-0.1969,-0.1683]
- Quadratic A: -0.3214, [-0.3411,-0.3022]
- Quadratic D: -0.1851, [-0.2000,-0.1705]

IEMOCAP, 10 speakers:

- all four Pitch+Rate point slopes remain negative;
- all four speaker-cluster CIs cross zero;
- pure within-speaker endpoint effects remain positive.

Correct wording: **strong confirmatory evidence on MSP; directionally consistent but low-powered hand-engineered external evidence on IEMOCAP.**

### Correct-center mechanism test — EXP-20260921-06

True-center Hybrid minus permuted-center Hybrid on raw MSP:

- Arousal +0.2955 CCC
- Dominance +0.2363 CCC

After within-speaker target centering:

- Arousal +0.00139
- Dominance +0.00100

Thus the benefit requires the center belonging to the correct speaker and target-level speaker structure; it is not a generic extra-dimension effect.

### Beyond pitch — EXP-20260921-14

F0 variability, pause ratio, voiced ratio, and their combined temporal/variability representation all show significant negative Arousal/Dominance target-reference slopes on MSP.

### Full six-attribute interpretable confirmation — EXP-20260922-01

Six attributes: pitch level, F0 variability, loudness, speaking rate, pause ratio, voiced ratio.

Speaker-cluster slopes:

- Linear A -0.5017
- Linear D -0.4044
- Quadratic A -0.5071
- Quadratic D -0.4092

All corresponding 95% CIs are fully below zero.

At lambda=0, Relative significantly beats Absolute in all four model-target cells. At lambda=1, Hybrid significantly beats Relative:

- Linear A +0.3593
- Linear D +0.2943
- Quadratic A +0.3758
- Quadratic D +0.3113

This is the strongest interpretable confirmation.

### WavLM confirmation — EXP-20260921-16/17

Frozen WavLM-large, IEMOCAP:

- layer 12 A slope ~-0.083
- layer 12 D ~-0.034
- layer 24 A ~-0.096
- layer 24 D ~-0.046

All four slopes are significantly negative under speaker-cluster bootstrap; all four lambda=0 effects favor Relative, and the preference reverses toward the raw endpoint.

Solver robustness (EXP-17): changing to high-precision LSQR changes primary slopes by at most 1.65e-6.

### Practical unlabeled enrollment

**EXP-20260921-18, WavLM:** moderate enrollment recovers the mechanism. At K=20 all four slope means are negative; at K=50 all four slope signs match the oracle. Very small K is not reliable.

**EXP-20260922-02, six-feature fixed pool:**

- Absolute A/D: 0.51969 / 0.42525 CCC
- K20 Hybrid: 0.55871 / 0.45702
- K50 Hybrid: 0.56800 / 0.46441
- Oracle Hybrid: 0.57243 / 0.46807

K20 Hybrid-minus-Absolute:

- A +0.0390
- D +0.0318

K50 remaining oracle gap:

- A 0.0044
- D 0.0037

K=20 is a practical operating point, **not** a universal or hard saturation threshold.

## Important negative results / boundaries

Do not silently drop these:

- **20-01 invalid:** RAVDESS fold/class-coverage metric defect; superseded by 20-02.
- **20-03 inconclusive:** simply appending Absolute vs Relative pitch to final-layer WavLM does not reproduce the large prosody-only reversal.
- **20-07 inconclusive:** Relative pitch does not generalize uniformly to natural categorical corpora; MSP is a counterexample.
- **20-13 reject:** confidence-aware routing underperforms the simpler router.
- **20-14/15 inconclusive:** categorical reference-frame effects are substantially classifier-dependent.
- **20-20 invalid:** K changed together with downstream sample pool; apparent saturation was confounded.
- **21-08 mixed:** better physical center estimation on IEMOCAP does not guarantee monotonic downstream mechanism recovery.
- **21-09 reject:** IEMOCAP loudness result fails when undocumented relative_db is replaced with physically interpretable RMS dB; do not use IEMOCAP loudness as primary evidence.
- **21-11 mixed:** Relative low-dimensional prosody only modestly suppresses speaker-ID; do not claim normalization removes speaker identity.
- **21-13 invalid:** deterministic LOSO folds were repeated under seed labels and incorrectly treated as independent; corrected session-level evidence is sensitivity-only and heterogeneous.

## Main-paper evidence stack

Prioritize:

1. EXP-20260920-02 — motivation
2. EXP-20260920-17 — target decomposition / hypothesis discovery
3. EXP-20260921-03/04 — controlled target intervention + nonlinear robustness
4. EXP-20260921-06 — correct-center permutation
5. EXP-20260921-14/15 — broader prosody + correct speaker-level inference
6. EXP-20260921-16/17 — WavLM mechanism + numerical robustness
7. EXP-20260922-01 — strongest interpretable confirmation
8. EXP-20260921-18 / EXP-20260922-02 — deployment with unlabeled enrollment

Earlier routing/classifier experiments are development history and boundary evidence, not the final paper’s main story.

## Statistical caution still open

EXP-20260922-02 reports paired fold/seed confidence intervals with n_pairs=15. Its fixed-pool point estimates are useful, but a **separately registered speaker-cluster robustness analysis** would align deployment inference with EXP-20260921-15 and EXP-20260922-01. Do not retroactively rewrite EXP-20260922-02.

## Paper status

The current COLING 2027 English draft in coling2027_draft/ already uses the corrected IEMOCAP speaker-cluster interpretation.

Avoid claims that:

- IEMOCAP hand-engineered slopes are independently significant;
- IEMOCAP loudness confirms the mechanism;
- K=20 is a universal saturation point;
- arbitrary few-shot enrollment reliably recovers WavLM reference matching;
- Relative prosody removes speaker identity;
- speaker normalization is universally beneficial or harmful.

## Next legal step

Highest-value optional robustness experiment: preregister a speaker-cluster inference analysis for the fixed-pool six-feature K-shot deployment result (EXP-20260922-02). Otherwise the experimental chain is mature enough to prioritize manuscript positioning, claim-evidence auditing, and reviewer-facing explanation.
