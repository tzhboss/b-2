# Experiment Report — EXP-20260921-16

Target-reference matching survives in frozen WavLM embedding space.

Speaker-relative WavLM embeddings are better than Absolute embeddings for pure within-speaker
Arousal/Dominance targets at both layer 12 and layer 24. As between-speaker target structure is
restored, the Relative advantage decreases systematically; all four layer-by-target slopes are
negative with speaker-cluster confidence intervals below zero.

The result is also speaker-consistent: all ten speakers show a negative Arousal slope at both
layers, and 8-9 of ten show a negative Dominance slope.

This directly weakens the interpretation that the earlier mechanism is a trivial consequence of
subtracting scalar handcrafted prosodic features. The same reference-frame interaction appears
when the reference operation is applied to 1024-dimensional learned speech embeddings.

A numerical-solver sensitivity is retained as the next check because default Ridge emitted
ill-conditioning warnings in the high-dimensional setting.
