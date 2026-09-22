# Experiment Report — EXP-20260921-16

Target-reference matching survives in learned speech representations.

Using frozen WavLM-large pooled embeddings, we construct:
- Absolute embeddings,
- speaker-relative embeddings,
- Hybrid embeddings.

On IEMOCAP, Relative-minus-Absolute CCC decreases monotonically as between-speaker Arousal or
Dominance target structure increases. The effect is significantly negative under speaker-cluster
bootstrap at both layer 12 and layer 24.

At the pure within-speaker endpoint, Relative embeddings significantly outperform Absolute
embeddings in all four layer-target conditions. At the raw-target endpoint, the preference reverses.

This shows that the reference-frame mechanism is not restricted to explicit hand-engineered
prosodic features.

Decision: pass.
