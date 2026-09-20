# CURRENT_STATE.md

## Phase

Cross-corpus reference-frame preference prediction completed and audited.

## Main valid findings

- Acoustic/class statistics predict the direction of Relative-versus-Absolute preference on a
  held-out corpus substantially better than attribute/emotion names alone.
- Stats-only LOCO resolved-sign accuracy is 79.1% with balanced accuracy 81.7%.
- Exact effect magnitude remains difficult, especially for extreme MSP neutral/rate cells.
- Leave-one-attribute-out is asymmetric and does not establish broad attribute-general prediction.
- Relative-versus-Absolute class separation and baseline-effect magnitude are the strongest
  mechanistically aligned predictors.

## Next legal step

Evaluate a held-out-corpus reference-frame router that uses the EXP-11 out-of-sample sign
predictions to choose Absolute or Relative for each attribute/emotion cell, comparing its achieved
F1 and regret against always-Absolute, always-Relative, labels-only routing, and oracle routing.
