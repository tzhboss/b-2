# Experiment Report — EXP-20260921-16

The target-reference effect is not confined to handcrafted prosodic features.

Speaker-relative WavLM embeddings outperform Absolute embeddings for pure within-speaker
Arousal/Dominance targets at both layer 12 and layer 24. As between-speaker target structure is
added, this advantage decreases monotonically and reverses for the raw target.

All four speaker-cluster slope confidence intervals are below zero.

This directly addresses the concern that the earlier result could be a trivial consequence of
subtracting handcrafted pitch/rate statistics: the same phenomenon appears in a 1024-dimensional
learned speech representation.

Decision: pass.
