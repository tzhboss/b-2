# Experiment Report — EXP-20260920-11

Simple acoustic/class statistics can predict the direction of Absolute-versus-Relative preference
on a held-out corpus substantially better than attribute/emotion identity alone.

The stats-only model reaches 79.1% resolved-sign accuracy and 81.7% balanced accuracy under
leave-one-corpus-out evaluation, compared with 58.1% / 52.0% for labels-only.

Adding categorical attribute/emotion identity slightly hurts relative to stats-only, suggesting
that corpus-general prediction is carried primarily by acoustic/statistical structure rather than
memorized label-specific preferences.

Exact effect magnitudes remain difficult, especially for extreme MSP cells such as neutral-rate.
Leave-one-attribute-out is also asymmetric: pitch can be predicted well from loudness/rate
statistics, rate is borderline, and loudness does not generalize.

Decision: mixed. Directional corpus-general predictability is supported; precise magnitude and
broad attribute-general prediction remain incomplete.
