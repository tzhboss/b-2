# Experiment Report — EXP-20260921-09

## Main Result

The IEMOCAP loudness-based target-reference result is not robust to a physically interpretable
loudness definition.

Replacing the undocumented source field relative_db with utterance RMS dB removes or reverses the
negative Arousal/Dominance target-reference slopes.

## Interpretation

The target-reference mechanism should not be claimed from IEMOCAP loudness. The source
relative_db field has unclear semantics and behaves very differently from RMS dB.

IEMOCAP remains useful for pitch/rate and for continuous VAD target structure, but loudness-based
evidence must be treated as a sensitivity result rather than a main claim.

## Decision

reject.
