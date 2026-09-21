# Closest Prior Work Matrix

| Work | Speaker handling | Target/task | Main assumption | What it establishes | What it does not establish relative to our work |
|---|---|---|---|---|---|
| Bone, Lee, Narayanan, Interspeech 2012; extended 2014 rule-based arousal work | Per-speaker neutral/global acoustic baselines; score utterance deviations relative to baseline | Continuous arousal | Raw acoustic values are hard to interpret across speakers; a speaker baseline enables relative arousal rating | Speaker-relative pitch/intensity/HF500 can provide interpretable and robust arousal ratings, including cross-corpus use | Does not compare Absolute vs Relative vs Hybrid as competing information frames; does not manipulate the target's between-speaker component; does not ask when speaker baseline should be preserved rather than normalized away |
| Busso/Mariooryad et al., 2013-2014 speaker/lexical variability compensation | Normalization/whitening to reduce speaker and lexical variability | Categorical emotion recognition | Speaker/lexical variability is nuisance that should be compensated | Speaker normalization can improve emotion classification and reduce nuisance variability | Treats speaker variation primarily as nuisance; no target-reference decomposition or condition under which baseline becomes task-relevant |
| Cao, Verma, Nenkova, 2015 speaker-sensitive ranking | Within-speaker ranking of emotion scores | Categorical emotion | Relative ranking within a speaker captures speaker expressivity | Speaker-sensitive ranking improves emotion classification | Speaker-relative operation is applied to classifier/ranker outputs, not explicit acoustic Absolute/Relative/Hybrid decomposition; no target-reference intervention |
| Gat et al., ICASSP 2022 | Adversarially suppress speaker characteristics | SER with SSL representations | Speaker information is a shortcut that harms generalization | Speaker normalization can improve SER and reduce speaker shortcuts | Does not model cases where speaker information is useful because the target contains between-speaker structure |
| Sridhar et al., Interspeech 2018; Sridhar & Busso 2022 | Analyze/adapt to speaker dependence | Valence/Arousal/Dominance | Some affect dimensions have stronger speaker-dependent traits | Speaker dependence changes generalization and personalization needs | Does not connect target speaker-dependence to the optimal acoustic reference frame |
| Shi, Li, Toda, Interspeech 2025 (SAMT) | Explicit speaker representation plus speaker-emotion disentanglement | SER | Speaker-specific information can be useful if modeled explicitly | Speaker-aware representations can improve SER in speaker-dependent and speaker-independent settings | Does not determine when speaker information should be preserved versus suppressed; no between/within target intervention |
| Gao et al., IEEE SPL 2026 (ASER) | Speaker-relative prosodic coding as intermediate attribute | Generative multi-task SER | Speaker-relative prosody is a reliable interpretable attribute | Relative prosodic tokens can improve explainable/attribute-aware SER | Uses speaker-relative prosody but does not compare its information trade-off against Absolute/Hybrid frames or vary target reference frame |
| Current work | Explicit Absolute, Relative, speaker-center, Hybrid representations; target decomposition and controlled target intervention | Categorical emotion/trait + continuous VAD | Speaker baseline can be nuisance or useful depending on target reference frame | Reference-frame preference shifts systematically as between-speaker target structure is manipulated; correct center identity matters only when target contains speaker-level structure; K-shot label-free reference estimation is feasible | Continuous external validation currently relies primarily on MSP + IEMOCAP; session-disjoint IEMOCAP evidence is heterogeneous |

## Main gap

The closest literature contains two apparently opposing traditions:

1. **Speaker normalization / invariance:** speaker variability is treated as nuisance or shortcut.
2. **Speaker-aware / personalization:** speaker information is explicitly modeled because it can help affect recognition.

The missing question is not whether speaker information can help or hurt in general.

The missing question is:

> **What determines whether stable speaker information is nuisance or task-relevant?**

Our experiments support one answer:

> **The usefulness of speaker baseline information depends on the reference frame of the target.**

This is operationalized by decomposing both representation and target into between-speaker and
within-speaker components, and then intervening on the target while holding the acoustic input and
model fixed.
