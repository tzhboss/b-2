# Experiment Report — EXP-20260920-12

A leave-one-corpus-out stats-only reference-frame router improves average speaker-balanced F1 over
both fixed policies and recovers most of the available oracle routing gain.

Across 75 cells, the router improves over always-Relative by +0.0096 F1 and reduces mean oracle
regret from 0.0157 to 0.0060. The gain is larger on resolved-effect cells (+0.0149).

The benefit is concentrated where fixed policies fail badly, especially MSP. Small negative gains
remain on MEAD and RAVDESS, motivating a conservative confidence-aware router whose threshold and
fallback are chosen strictly from training corpora.

Decision: mixed.
