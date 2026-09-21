# 3. Method

## 3.1 Prosodic reference frames

For speaker s and utterance u, let an acoustic prosodic feature be

x_(s,u) = mu_s + delta_(s,u),

where mu_s is a stable speaker reference and delta_(s,u) is the utterance's deviation from that
reference.

We compare three representation families:

- **Absolute:** x_(s,u)
- **Relative:** x_(s,u) - mu_s
- **Hybrid:** [x_(s,u) - mu_s, mu_s]

For the main confirmatory VAD experiment, the acoustic dimensions are Pitch and Speaking Rate,
which have clear semantics in both MSP-Podcast and IEMOCAP. Pitch is represented on a semitone/log
scale, and rate is log transformed. The speaker reference mu_s is the per-speaker marginal median
of each acoustic dimension in the diagnostic oracle analyses.

Loudness is excluded from the cross-corpus confirmatory experiment because the available IEMOCAP
field relative_db has undocumented semantics and does not agree with a physically interpretable
RMS-dB reconstruction in our sensitivity analysis.

## 3.2 Target reference frames

For each VAD target y_(s,u), define the speaker mean ybar_s and global mean ybar. The observed target
can be written as a sum of within-speaker deviation and between-speaker offset.

To manipulate target reference frame without altering acoustic inputs, we define

y_lambda(s,u) =
    ybar
    + [y_(s,u) - ybar_s]
    + lambda [ybar_s - ybar],

where lambda is in {0, 0.25, 0.5, 0.75, 1}.

At lambda = 0, all between-speaker target offsets are removed and the target contains only
within-speaker variation around the global mean. At lambda = 1, the construction recovers the
original raw target exactly. Intermediate lambda values continuously reintroduce the
between-speaker component.

This intervention keeps utterances, acoustic features, speakers, train/test partitions, and model
family fixed. Only the reference frame of the target changes.

## 3.3 Primary hypothesis

Let

Delta(lambda) = CCC_Relative(lambda) - CCC_Absolute(lambda).

The target-reference matching hypothesis predicts

d Delta(lambda) / d lambda < 0

for affect dimensions whose acoustic deviations track within-speaker state while speaker acoustic
centers carry information about between-speaker target structure.

A complementary diagnostic is Hybrid-minus-Relative utility. Stable speaker centers should become
more useful as the target contains more between-speaker information, although nonlinear models may
recover part of that structure implicitly from Absolute or Relative features.

## 3.4 Datasets

### MSP-Podcast

MSP-Podcast provides large-scale natural speech with continuous Valence, Arousal, and Dominance
ratings. Our main MSP analyses use speaker-disjoint folds and speaker-balanced evaluation. The
dataset contains enough speakers to estimate between-speaker target structure, perform permutation
interventions, and evaluate label-free K-shot enrollment.

### IEMOCAP

IEMOCAP provides continuous dimensional ratings together with acted and improvised dyadic speech.
The main cross-corpus confirmatory experiment uses Pitch and Speaking Rate only. IEMOCAP has ten
speakers, which provides an independent dataset but limited power for speaker-level inferential
tests. A stricter leave-one-session-out sensitivity analysis shows substantial session
heterogeneity and is treated as a limitation rather than confirmatory evidence.

### ESD, MEAD, and RAVDESS

These datasets are used for the categorical state-versus-trait motivation. We compare reference
frames for five-class emotion classification and gender classification using low-dimensional
prosody.

## 3.5 Models and evaluation

The confirmatory target-intervention analysis uses two fixed model families:

1. Linear Ridge regression.
2. Quadratic Ridge regression, implemented by standardized degree-2 polynomial expansion followed
   by re-standardization and Ridge regression.

No corpus-specific hyperparameter tuning is used for the target-reference hypothesis.

Continuous affect is evaluated with speaker-balanced Concordance Correlation Coefficient (CCC).
Categorical tasks use Macro-F1. Speaker-disjoint splits are used unless otherwise stated.

## 3.6 Speaker-center permutation intervention

To test whether Hybrid gains arise from the correct speaker reference rather than merely additional
center-valued dimensions, we keep each utterance's Relative features unchanged while reassigning
speaker centers across speakers using a derangement.

The intervention preserves the marginal distribution of center values but destroys
speaker-center identity. Separate derangements are applied within training and test speakers.

If the correct center is useful because it is aligned with speaker-level target structure, then
true-center Hybrid should outperform permuted-center Hybrid for raw targets. The effect should
collapse after the target is centered within speaker.

## 3.7 Label-free K-shot reference estimation

Oracle speaker references assume access to many utterances. For deployment analysis, we estimate
the acoustic center from K unlabeled enrollment utterances from each speaker. Enrollment utterances
are excluded from downstream train/test rows.

A corrected fixed-pool experiment reserves the same 50 enrollment candidates for every K and keeps
all downstream rows identical across K. We compare K in {1, 2, 5, 10, 20, 50} with a full
marginal-speaker oracle.

## 3.8 Representation analysis with WavLM

To test whether a modern self-supervised speech encoder already retains information corresponding
to multiple prosodic reference frames, we probe frozen WavLM-large hidden layers for Absolute
pitch, Relative pitch, and implied speaker baseline pitch.

This analysis addresses decodability, not invariance. High decodability of multiple targets means
that the encoder preserves information related to multiple frames; it does not imply that explicit
Relative features will necessarily improve an emotion classifier on top of WavLM.
