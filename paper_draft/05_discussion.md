# 5. Discussion

## 5.1 Normalization is an information transformation

Speaker normalization is often discussed as if it removes unwanted variation while leaving the
task-relevant signal intact. Our results support a different interpretation.

Subtracting a speaker baseline is an information transformation. It suppresses stable
between-speaker information and emphasizes within-speaker deviation. Whether this helps depends on
what the target asks the model to predict.

The target intervention makes this dependence visible. With the acoustic observations held fixed,
changing only the amount of between-speaker structure in the target systematically changes the
Relative-versus-Absolute performance difference. This is the central empirical support for the
reference-frame matching account.

## 5.2 Reconciling speaker invariance and personalization

Two apparently conflicting directions coexist in SER.

Speaker-invariant approaches attempt to suppress speaker information because inter-speaker
variation can become a shortcut and hurt unseen-speaker generalization. Personalization approaches,
by contrast, explicitly condition on speaker characteristics or speaker enrollment.

Our results suggest that neither objective is universally preferable.

For a within-speaker state target, stable speaker information is nuisance by construction.
Relative prosody is therefore well matched to the target.

For a population-level target that contains stable between-speaker structure, the speaker baseline
can become predictive information. Removing it can reduce performance, while exposing it
separately in a Hybrid representation can help.

Thus speaker invariance and personalization can be understood as different choices along the same
reference-frame axis.

## 5.3 Relation to prior speaker-normalized arousal work

The strongest historical precursor is Bone, Lee, and Narayanan (2012), who already demonstrated
that prosodic features scored relative to a speaker's neutral baseline can support robust
cross-corpus arousal rating, including with limited neutral reference data. Busso et al. (2013) and
Mariooryad and Busso (2014) further established speaker normalization and compensation as useful
SER strategies.

The present work should therefore not be positioned as introducing relative prosody or
speaker-baseline enrollment.

The distinction is explanatory rather than procedural. Earlier work primarily asks how to obtain
better emotion recognition by compensating speaker variability. We ask when that compensation
should help or hurt. The controlled lambda intervention manipulates the target reference frame
directly and shows that representation preference moves with it.

## 5.4 Why Arousal and Dominance are clearer than Valence

Low-dimensional prosody strongly supports the proposed mechanism for Arousal and Dominance, but not
for Valence.

On MSP, Valence has lower speaker-level acoustic predictability than Arousal/Dominance in our
simple prosodic feature space, and both raw and within-speaker Valence CCC remain low. This agrees
with the broader SER literature that finds Valence harder to infer from acoustics and more
dependent on speaker, lexical, contextual, or semantic information.

Valence should therefore be treated as a boundary condition rather than forced into the same
low-dimensional prosody explanation.

## 5.5 Why the IEMOCAP effect is smaller than MSP

MSP and IEMOCAP differ substantially in target structure. MSP Arousal and Dominance show large
between-speaker variance fractions, whereas IEMOCAP's are much smaller. Consequently, the
Relative-minus-Absolute slope is much steeper on MSP.

This difference is predicted by the reference-frame account: there is simply less speaker-level
target structure in IEMOCAP for Absolute/Hybrid representations to exploit.

The stricter session-disjoint sensitivity also shows that the IEMOCAP effect is heterogeneous over
only five dyadic sessions. The IEMOCAP evidence should therefore be described as an external
replication of direction under the main speaker-disjoint protocol, not as proof of universal
session-level consistency.

## 5.6 Feature semantics matter

A key sensitivity result concerns loudness.

The available IEMOCAP relative_db field behaves differently from an RMS-dB reconstruction, and the
loudness-based target-reference result does not survive this substitution. This demonstrates a
general methodological lesson: a reference-frame study is only meaningful if the physical meaning
of the feature and its baseline are clearly defined.

For this reason, the main cross-corpus result uses only Pitch and Speaking Rate, whose definitions
are transparent in both corpora.

## 5.7 Deployment implications

The oracle decomposition is useful for mechanism analysis but cannot be assumed at inference time.

The K-shot experiments show that useful speaker references can be estimated from unlabeled
enrollment audio. On the corrected MSP fixed-pool analysis, approximately 20 utterances recover
most of the practical Hybrid benefit.

This result suggests that reference-frame-aware systems need not require emotion labels or a
manually collected neutral state. However, the amount and composition of enrollment data remain
corpus-dependent. IEMOCAP's non-monotonic K-shot mechanism recovery illustrates that more accurate
physical center estimation does not guarantee monotonic downstream gains.

## 5.8 Implications for representation learning

The WavLM probes show that Absolute pitch, Relative pitch, and implied speaker baseline are all
strongly decodable from hidden representations, with different layer profiles.

This suggests that modern SSL encoders do not simply become speaker-invariant in a binary sense.
They can retain multiple overlapping forms of state and trait information.

A useful next step for representation learning is therefore not merely to ask whether speaker
information is present, but whether downstream readouts emphasize the reference frame appropriate
for the target.

## 5.9 Limitations

The present study has several important limitations.

1. Continuous-VAD external validation is limited to MSP-Podcast and IEMOCAP.
2. IEMOCAP contains only ten speakers and five dyadic sessions; strict session-level replication is
   heterogeneous.
3. The main interpretable acoustic space is intentionally low-dimensional. More complex prosodic
   contours, voice quality, lexical content, and multimodal cues may interact with reference frame
   differently.
4. The marginal speaker center is a diagnostic oracle. K-shot experiments reduce this deployment
   gap but do not solve reference estimation for all settings.
5. The target intervention is a controlled statistical construction. It establishes a mechanism
   within the observed dataset but does not imply that naturally occurring applications will have
   a known lambda.
6. Gender is used as a strong stable-trait probe, but speaker-ID results show that Relative prosody
   only modestly suppresses full speaker identity.
7. The study does not establish that every affect dimension or every corpus follows the same effect
   magnitude.

## 5.10 Broader methodological implication

SER evaluation often treats a label such as Arousal as if its meaning were independent of
population structure. Our results suggest that evaluation should state whether the intended target
is:

- population-level absolute affect,
- deviation from a person's own baseline,
- or a combination of both.

These targets are not interchangeable, and they can prefer different acoustic representations.

The main conclusion is therefore not a recommendation to normalize or not normalize. It is a
recommendation to make the reference frame explicit.
