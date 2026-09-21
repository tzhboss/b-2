# Abstract

Speaker normalization is commonly used in speech emotion recognition (SER) to suppress
speaker-dependent variability, implicitly treating stable speaker characteristics as nuisance.
We argue that this assumption is task-dependent: whether speaker baseline information should be
removed depends on the reference frame of the prediction target.

We formulate prosodic reference frames by decomposing acoustic features into stable speaker
baselines and within-speaker deviations, and decompose continuous affect targets analogously into
between-speaker and within-speaker components. We then introduce a controlled target intervention
that continuously varies the strength of between-speaker target structure while keeping acoustic
inputs, train/test splits, and model family fixed.

Using semantically matched Pitch + Speaking Rate features, we find that
Relative-minus-Absolute CCC decreases significantly as between-speaker Arousal/Dominance target
structure increases in both MSP-Podcast and IEMOCAP, under both linear and quadratic Ridge models.
At the pure within-speaker endpoint, Relative outperforms Absolute in every confirmatory
corpus-by-target-by-model cell. A speaker-center permutation intervention further shows that the
large Hybrid benefit on raw MSP Arousal/Dominance depends on assigning the correct center to the
correct speaker, whereas the effect nearly disappears after within-speaker target centering.
Finally, label-free enrollment experiments show that roughly 20 utterances recover most of the
practical speaker-reference utility on a fixed MSP downstream pool.

These results support a conditional view of normalization: stable speaker information can be
nuisance or useful signal depending on target structure. We therefore propose that representation
reference frame should be matched to target reference frame rather than assuming that speaker
invariance is universally desirable.
