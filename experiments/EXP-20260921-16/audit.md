# Experiment Audit — EXP-20260921-16

## Extraction integrity

- 10,039 / 10,039 IEMOCAP utterances embedded.
- Three parquet shards: 3,347 + 3,346 + 3,346 utterances.
- No duplicate sample IDs.
- No non-finite embeddings.
- Frozen microsoft-wavlm-large.
- Mean-pooled hidden layers 12 and 24, 1024 dimensions each.
- Extraction processes all exited with code 0.
- Probe process exited with code 0.
- Runtime script/config hashes match.

## Speaker-cluster target-reference slopes

Relative-minus-Absolute CCC slope versus lambda:

Layer 12:
- Arousal: -0.08294, 95% CI [-0.13092, -0.03591].
- Dominance: -0.03406, [-0.05835, -0.01180].

Layer 24:
- Arousal: -0.09578, 95% CI [-0.14289, -0.04818].
- Dominance: -0.04555, [-0.07067, -0.01810].

All four speaker-cluster CIs are strictly below zero despite only ten speakers.

## Pure within-speaker endpoint

Relative-minus-Absolute CCC at lambda=0:

- Layer 12 Arousal: +0.04046, 95% CI [+0.01850, +0.06536].
- Layer 12 Dominance: +0.01483, [+0.00784, +0.02155].
- Layer 24 Arousal: +0.04659, [+0.02865, +0.06627].
- Layer 24 Dominance: +0.02507, [+0.01519, +0.03483].

All four endpoints significantly favor Relative.

## Shape

As lambda increases from 0 to 1, Relative-minus-Absolute moves from positive to negative in every
layer-target cell.

Layer 12:
- Arousal: +0.0405 -> -0.0423.
- Dominance: +0.0148 -> -0.0192.

Layer 24:
- Arousal: +0.0466 -> -0.0490.
- Dominance: +0.0251 -> -0.0204.

## Registered criteria

- SSL direction support: supported.
- Within-speaker endpoint support: supported 4/4.
- Strong SSL replication: supported 4/4 slope CIs below zero.
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: pass.

Target-reference matching extends from explicit prosodic attributes to frozen WavLM embedding
space. This directly weakens the interpretation that the mechanism is a handcrafted-feature
artifact.
