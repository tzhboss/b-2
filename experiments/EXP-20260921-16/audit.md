# Experiment Audit — EXP-20260921-16

## Extraction integrity

- Initial extraction attempt failed safely before producing embeddings because the then-used
  environment had Torch <2.6 and current Transformers refused to load the legacy model binary
  under the CVE-2025-32434 safety check.
- No safety check was bypassed.
- Extraction was rerun in the historical WavLM environment used by EXP-20260920-03/04:
  Torch 2.7.1+cu126, Transformers 4.57.6.
- Frozen checkpoint: /data/lc/models/microsoft-wavlm-large.
- Extracted layers fixed before result inspection: 12 and 24.
- All three parquet shards completed with exit code 0.
- Coverage: 10,039 / 10,039 IEMOCAP utterances.
- Unique sample IDs: 10,039 / 10,039.
- Embedding shape: (N, 2, 1024).
- Layer order: [12, 24].
- All embedding values finite.

## Probe integrity

- Five fixed speaker-disjoint folds.
- Speaker center constructed as coordinate-wise full-speaker embedding median.
- Full-speaker center is explicitly diagnostic/oracle, not a deployment assumption.
- Target intervention matches the earlier explicit-prosody protocol.
- Uncertainty uses speaker-cluster bootstrap with 5,000 replicates.
- All OOF targets and Absolute/Relative/Hybrid predictions are finite.
- High-dimensional Ridge emitted ill-conditioning warnings, especially relevant to Hybrid
  concatenation. No numerical failure occurred. The primary registered criterion is
  Relative-minus-Absolute and should be emphasized over small Hybrid differences.

## Speaker-cluster results

Relative-minus-Absolute slope versus lambda:

Layer 12:
- Arousal: -0.08294, 95% CI [-0.13092, -0.03591].
- Dominance: -0.03406, [-0.05835, -0.01180].

Layer 24:
- Arousal: -0.09578, 95% CI [-0.14289, -0.04818].
- Dominance: -0.04555, [-0.07067, -0.01810].

All four slope CIs exclude zero below.

Pure within-speaker endpoint, lambda=0:

Layer 12:
- Arousal Relative-Absolute: +0.04046, 95% CI [0.01850, 0.06536].
- Dominance: +0.01483, [0.00784, 0.02155].

Layer 24:
- Arousal: +0.04659, 95% CI [0.02865, 0.06627].
- Dominance: +0.02507, [0.01519, 0.03483].

All four endpoint CIs exclude zero above.

Raw target endpoint, lambda=1:
- Layer 12 Arousal: -0.04235; Dominance: -0.01917.
- Layer 24 Arousal: -0.04903; Dominance: -0.02040.

Thus the representation preference reverses across the target-reference continuum.

## Hybrid diagnostic

At lambda=0 Hybrid and Relative are almost identical.
At lambda=1 Hybrid recovers part of the loss from removing speaker-center information.

Examples:
- Layer 12 Arousal lambda=1: Absolute 0.7210, Relative 0.6786, Hybrid 0.7024 CCC.
- Layer 24 Arousal lambda=1: Absolute 0.7091, Relative 0.6600, Hybrid 0.6879.

Given the Ridge conditioning warnings, treat these Hybrid magnitudes as supportive diagnostics,
not the primary inferential claim.

## Registered criteria

- Negative SSL target-reference direction in all four layer-target cells: supported.
- Positive lambda=0 Relative advantage in at least 3/4 cells: supported (4/4).
- Strong SSL replication with at least 3/4 slope CIs below zero: supported (4/4).
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: pass.

The target-reference mechanism therefore generalizes from interpretable explicit prosody to frozen
WavLM embedding space.
