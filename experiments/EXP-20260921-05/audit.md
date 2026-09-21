# Experiment Audit — EXP-20260921-05

## Runtime integrity

- Initial execution failed before producing scientific outputs because loudness was mapped to a
  non-existent loudness_* column prefix.
- The implementation-only mapping bug was fixed in commit
  f9f28ba3c3d5cdb00dc33c60ed4d62b8c5b7fb71 without changing protocol, data, hypotheses, or thresholds.
- The rerun used immutable /tmp script/config copies whose SHA256 values match the fixed source.
- 4,050 / 4,050 fold metric rows complete.
- 90 / 90 effect curves complete.
- 36 / 36 slope tests complete.
- No metric NaNs.

## Relative-minus-Absolute slopes versus lambda

MSP:
- Pitch Arousal: -0.2952; Dominance: -0.1681.
- Loudness Arousal: -0.3165; Dominance: -0.2991.
- Rate Arousal: -0.0253; Dominance: -0.0133.
All are significantly negative.

IEMOCAP:
- Pitch Arousal: -0.0190, significant; Dominance: -0.0003, unresolved.
- Loudness Arousal: -0.0182; Dominance: -0.0131, both significant.
- Rate Arousal: -0.0085; Dominance: -0.0123, both significant.

## Registered Criteria

- Pitch coupling: supported.
- Loudness coupling: supported.
- Rate coupling: supported.
- Cross-attribute heterogeneity: supported.

MSP shows a large slope range across attributes:
- Arousal range: 0.2912.
- Dominance range: 0.2858.

IEMOCAP slopes are much more compressed:
- Arousal range: 0.0105.
- Dominance range: 0.0128.

## Verdict

- Validity: valid.
- Decision: pass.
