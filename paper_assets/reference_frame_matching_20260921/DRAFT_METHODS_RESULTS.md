# Draft Methods + Results

## Methods

### Problem formulation

We study prosodic normalization as a reference-frame choice. For speaker s and utterance u, an
utterance-level acoustic feature is decomposed as

x_su = μ_s + δ_su,

where μ_s is a stable speaker-specific acoustic center and δ_su is a within-speaker deviation.

For continuous affect, we analogously write

y_su = ȳ_s + ε_su,

where ȳ_s is the speaker-level target mean and ε_su is the within-speaker target residual.

We compare three representation families:

- Absolute: x_su
- Relative: x_su − μ_s
- Hybrid: [x_su − μ_s, μ_s]

The core hypothesis is that the preferred representation depends on the target reference frame.

### Categorical state-versus-trait pilot

We first evaluate explicit prosodic features on ESD, MEAD, and RAVDESS. We construct Absolute,
Relative, and Hybrid representations from pitch, loudness, and speaking rate, and evaluate
5-class emotion recognition and gender classification under matched folds.

The purpose of this experiment is motivational rather than confirmatory: it asks whether changing
the acoustic reference frame redistributes information useful for a state-like target (emotion)
versus a stable speaker trait (gender).

### Continuous affect datasets

We use MSP-Podcast for large-scale continuous Valence, Arousal, and Dominance prediction and
IEMOCAP for external validation.

For MSP, the dimensional targets are human SAM ratings on a 1-7 scale. For IEMOCAP, we use the
continuous dimensional annotations EmoVal, EmoAct, and EmoDom.

### Acoustic features

The clean cross-corpus confirmatory experiment uses only features with aligned and interpretable
semantics in both corpora:

- Pitch:
  - MSP: 12 log2(f0_median_hz)
  - IEMOCAP: 12 log2(pitch_mean)
- Speaking rate:
  - MSP: log(phoneme_articulation_rate)
  - IEMOCAP: log(speaking_rate)

IEMOCAP loudness is excluded from the primary confirmatory evidence because the source field
relative_db has undocumented semantics and failed robustness against a physically interpretable
RMS-dB replacement.

Speaker centers are computed as full-speaker marginal medians in the mechanism experiments.
These oracle centers are diagnostic quantities, not a deployment assumption.

### Target decomposition

For each continuous target, we estimate the speaker mean ȳ_s and define the within-speaker residual

ε_su = y_su − ȳ_s.

This allows us to compare raw population-level prediction with explicitly within-speaker prediction.

### Controlled target-reference intervention

To directly vary the target reference frame while keeping acoustic inputs and models fixed, we
construct

y_su(λ) = ȳ + ε_su + λ(ȳ_s − ȳ),

with λ ∈ {0, 0.25, 0.5, 0.75, 1}.

λ=0 removes the between-speaker component while preserving within-speaker variation.
λ=1 recovers the original raw target.

For each λ we train the same models on the same acoustic inputs and evaluate the change in

CCC(Relative) − CCC(Absolute).

A negative slope versus λ means Relative becomes less favorable as between-speaker target
structure increases.

### Models and evaluation

The primary models are:

- Linear Ridge regression
- Quadratic Ridge regression using degree-2 polynomial features after train-only standardization

We use speaker-disjoint folds, equal total train/evaluation weight per speaker, fixed regularization,
and no target-specific or corpus-specific hyperparameter tuning.

Continuous affect is evaluated using speaker-balanced concordance correlation coefficient (CCC).
Categorical tasks use Macro-F1.

### Speaker-center permutation intervention

To test whether Hybrid utility depends on the correct speaker center rather than merely additional
features, we keep the Relative features unchanged and replace each speaker's center with another
speaker's center via a within-split derangement. The center marginal distribution is preserved while
speaker-center identity is destroyed.

### Label-free K-shot enrollment

For deployment, we estimate a new speaker's acoustic center from K unlabeled enrollment utterances.
Enrollment utterances are excluded from downstream train/test rows. We use the per-dimension median
of pitch, loudness, and log-rate and compare K-shot Hybrid against Absolute and a diagnostic
marginal-oracle Hybrid.

The corrected fixed-pool study reserves 50 enrollment utterances per speaker and uses identical
downstream rows for every K ∈ {1, 2, 5, 10, 20, 50}.

### Frozen WavLM analysis

We additionally probe frozen WavLM-large hidden layers for decodability of Absolute pitch,
speaker-relative pitch, and implied speaker baseline. This analysis asks whether modern SSL
representations retain information corresponding to multiple prosodic reference frames.

---

## Results

### 1. Relative pitch helps categorical emotion but removes trait information

Across ESD, MEAD, and RAVDESS, Relative pitch consistently improves 5-class emotion Macro-F1 over
Absolute pitch while strongly reducing gender Macro-F1.

Relative-minus-Absolute pitch Macro-F1:

- ESD emotion: +0.0908; gender: -0.1330
- MEAD emotion: +0.0904; gender: -0.3790
- RAVDESS emotion: +0.0586; gender: -0.2299

This establishes that normalization redistributes task-relevant information rather than uniformly
improving representation quality.

### 2. MSP raw VAD contains substantial between-speaker structure

The ICC-like between-speaker fractions in MSP are approximately:

- Valence: 0.213
- Arousal: 0.418
- Dominance: 0.346

A stable prosodic baseline alone predicts speaker-level target means with CCC:

- Arousal: 0.482
- Dominance: 0.441
- Valence: 0.022

Thus raw Arousal and Dominance contain large speaker-level components that simple stable acoustic
statistics can predict.

### 3. Once the target is centered within speaker, Relative becomes strongly preferable

For within-speaker residual targets, Relative-minus-Absolute all-prosody CCC is:

- Arousal: +0.1306, 95% CI [0.1282, 0.1326]
- Dominance: +0.0921, [0.0894, 0.0945]
- Valence: +0.0049

Adding the speaker baseline back to Relative after target centering provides essentially no gain:

- Arousal: +0.00049
- Dominance: +0.00029

This directly supports the idea that speaker baseline utility depends on target reference frame.

### 4. Controlled target intervention confirms target-reference matching

The cleanest confirmatory experiment uses only Pitch+Rate in both MSP and IEMOCAP and evaluates
both Linear and Quadratic Ridge.

Slope of CCC(Relative) − CCC(Absolute) versus λ:

Linear Ridge:
- MSP Arousal: -0.3181, 95% CI [-0.3264, -0.3082]
- MSP Dominance: -0.1821, [-0.1878, -0.1750]
- IEMOCAP Arousal: -0.02245, [-0.03772, -0.00838]
- IEMOCAP Dominance: -0.01089, [-0.01684, -0.00348]

Quadratic Ridge:
- MSP Arousal: -0.3207, 95% CI [-0.3293, -0.3103]
- MSP Dominance: -0.1846, [-0.1906, -0.1771]
- IEMOCAP Arousal: -0.02646, [-0.04310, -0.01134]
- IEMOCAP Dominance: -0.01752, [-0.02440, -0.00994]

All eight confirmatory slopes are significantly negative.

At λ=0, Relative is better than Absolute in every Arousal/Dominance corpus-target-model cell:

- MSP Linear: +0.1389 Arousal, +0.0727 Dominance
- MSP Quadratic: +0.1389 Arousal, +0.0746 Dominance
- IEMOCAP Linear: +0.1363 Arousal, +0.0458 Dominance
- IEMOCAP Quadratic: +0.1551 Arousal, +0.0486 Dominance

The central mechanism therefore survives corpus change, semantically matched features, and
nonlinear feature access.

### 5. Correct speaker-center identity is necessary for raw Hybrid utility

We keep Relative features fixed and replace the appended speaker center with another speaker's
center.

On MSP raw targets, true center minus permuted center yields:

- Arousal: +0.2955 CCC, 95% CI [0.2856, 0.3039]
- Dominance: +0.2363, [0.2285, 0.2436]

After target centering, the same effect collapses to:

- Arousal: +0.00139
- Dominance: +0.00100

Thus the Hybrid gain is not caused by adding generic center-valued dimensions; it depends on the
correct speaker prior and disappears when the target no longer contains speaker-level structure.

IEMOCAP shows much smaller raw center-identity effects, consistent with its weak between-speaker
target structure.

### 6. About 20 unlabeled enrollment utterances recover most practical speaker-reference utility

In the corrected fixed-downstream-pool MSP experiment, K-shot Hybrid CCC is:

Arousal:
- K1: 0.4795
- K10: 0.4871
- K20: 0.4903
- K50: 0.4946

Dominance:
- K1: 0.3793
- K10: 0.3837
- K20: 0.3839
- K50: 0.3840

At K=20, K-shot Hybrid is only 0.00427 CCC below the marginal oracle for Arousal and is effectively
identical for Dominance (+0.00036 relative to oracle).

Therefore deployment does not require affect labels or a full speaker history to recover most of
the usable speaker-reference benefit.

### 7. WavLM retains multiple reference frames

Frozen WavLM-large representations preserve highly decodable information about Absolute pitch,
Relative pitch, and implied speaker baseline.

For Relative pitch, mean R² declines from earlier/middle to late layers:

- ESD: 0.8825 in layers 5-12 to 0.8114 in layers 21-24
- MEAD: 0.8462 to 0.7609
- RAVDESS: 0.8766 to 0.7754

Implied speaker baseline remains very strongly decodable throughout the network.

This shows that SSL representations do not simply become speaker-invariant. Instead, reference-frame
information remains available but is reorganized with depth.

### 8. Boundaries

The mechanism is strongest for Arousal and Dominance. Valence is weak under these low-dimensional
prosodic features and should not be overinterpreted.

IEMOCAP loudness is excluded from the main claim because replacing relative_db with RMS dB removes
or reverses the loudness-specific mechanism.

A stricter IEMOCAP leave-one-session-out sensitivity analysis has only five independent sessions and
shows substantial session heterogeneity. It should be treated as a limitation rather than
confirmatory evidence.

Speaker-relative low-dimensional prosody only modestly reduces speaker-ID decodability, even though
gender suppression is strong. We therefore avoid claiming that Relative prosody universally removes
speaker identity.

### Learned-representation target-reference intervention

We additionally construct reference frames directly in frozen WavLM-large embedding space on
IEMOCAP. We extract attention-mask-aware temporal mean embeddings from hidden layers 12 and 24.
For each layer, the Absolute representation is the utterance embedding, the speaker center is the
coordinate-wise median embedding over that speaker's utterances, and the Relative representation
subtracts this center from the utterance embedding. Hybrid concatenates Relative and center.

We then repeat the same target-reference lambda intervention under fixed speaker-disjoint folds.
Uncertainty is computed with speaker-cluster bootstrap.

### 9. Target-reference matching also appears in WavLM embedding space

The WavLM experiment produces a direct preference reversal.

Layer 12:
- Arousal slope: -0.0829, 95% CI [-0.1309, -0.0359].
- Dominance slope: -0.0341, [-0.0583, -0.0118].

Layer 24:
- Arousal slope: -0.0958, 95% CI [-0.1429, -0.0482].
- Dominance slope: -0.0455, [-0.0707, -0.0181].

At lambda=0, Relative-minus-Absolute is positive in every cell:
+0.0405/+0.0148 at layer 12 and +0.0466/+0.0251 at layer 24 for
Arousal/Dominance respectively.

At lambda=1, all four differences reverse sign:
-0.0423/-0.0192 at layer 12 and -0.0490/-0.0204 at layer 24.

Thus the core phenomenon is not restricted to interpretable low-dimensional prosody. A
speaker-relative transformation of a learned speech representation becomes less appropriate as
the target contains more between-speaker structure.
