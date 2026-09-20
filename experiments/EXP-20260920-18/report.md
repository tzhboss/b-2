# Experiment Report — EXP-20260920-18

## Main Result

The VAD reference-frame mechanisms are not artifacts of noisy human labels.

Among held-out utterances whose VAD ratings have low annotator disagreement, raw Arousal and
Dominance still strongly benefit from restoring stable speaker baseline information. Conversely,
after removing the speaker VAD mean from the target, speaker-relative prosody still strongly
outperforms Absolute prosody.

## Label Reliability

Prediction quality does not decline monotonically with disagreement for every target. The clearest
reliability gradient occurs for within-speaker Arousal residuals, where low-disagreement CCC is
about 0.071 higher than high-disagreement CCC using the best low-disagreement representation.

## Interpretation

The core decomposition result is robust to human label agreement:
- raw population-level VAD rewards stable speaker information;
- within-speaker affect deviation rewards speaker-relative prosody.

## Decision

mixed.
