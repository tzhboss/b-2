# Experiment Audit — EXP-20260921-16

## Runtime integrity and coverage

- Frozen local microsoft-wavlm-large checkpoint.
- Layers fixed before result inspection: 12 and 24.
- IEMOCAP coverage: 10,039 / 10,039 utterances.
- 3 embedding shards, 10,039 unique sample IDs, no duplicates.
- Embedding shape: [utterance, 2 layers, 1024].
- No non-finite embedding values.
- Extraction and probe completed with exit code 0.
- Runtime script/config hashes match source.

## Speaker-cluster target-reference results

Relative-minus-Absolute CCC slope versus lambda:

Layer 12:
- Arousal: -0.0829, 95% speaker-cluster CI [-0.1309, -0.0359].
- Dominance: -0.0341, [-0.0583, -0.0118].

Layer 24:
- Arousal: -0.0958, 95% CI [-0.1429, -0.0482].
- Dominance: -0.0455, [-0.0707, -0.0181].

All four slope CIs are strictly below zero despite only ten speakers.

## Pure within-speaker endpoint

Relative-minus-Absolute at lambda=0:

- Layer 12 Arousal: +0.0405, CI [+0.0185, +0.0654].
- Layer 12 Dominance: +0.0148, [+0.0078, +0.0215].
- Layer 24 Arousal: +0.0466, [+0.0286, +0.0663].
- Layer 24 Dominance: +0.0251, [+0.0152, +0.0348].

All four are significantly positive.

## Per-speaker diagnostic

Slope sign:
- Layer 12 Arousal: 10/10 speakers negative.
- Layer 12 Dominance: 9/10 negative.
- Layer 24 Arousal: 10/10 negative.
- Layer 24 Dominance: 8/10 negative.

Lambda=0 Relative-minus-Absolute sign:
- Arousal: 10/10 positive at both layers.
- Dominance: 9/10 positive at both layers.

Thus the aggregate effect is not driven by a single speaker.

## Hybrid diagnostic

At lambda=0, Hybrid-minus-Relative is approximately zero.
At lambda=1, speaker-weighted Hybrid-minus-Relative is positive for the principal Arousal settings
and generally increases relative to the within-speaker endpoint. This is consistent with stable
speaker-center information becoming useful as between-speaker target structure is restored.

## Numerical note

scikit-learn emitted ill-conditioned-matrix warnings for some Ridge solves in the high-dimensional
embedding setting. Predictions and embeddings remained finite. A solver-stability sensitivity
check should be performed before treating the exact magnitudes as final; the sign pattern is
currently strong and speaker-consistent.

## Registered criteria

- SSL target-reference direction: supported.
- Within-speaker endpoint support: supported (4/4).
- Strong SSL replication: supported (4/4 speaker-cluster slope CIs below zero).
- Rejection criterion: not met.

## Verdict

- Validity: valid, pending numerical-solver sensitivity for exact-magnitude robustness.
- Decision: pass.

This extends target-reference matching from explicit low-dimensional prosody to learned frozen
speech representations.
