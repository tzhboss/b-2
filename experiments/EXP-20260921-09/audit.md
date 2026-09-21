# Experiment Audit — EXP-20260921-09

## Runtime integrity

- Source/runtime script and config SHA256 values match.
- Immutable /tmp execution used.
- 10,039 / 10,039 IEMOCAP utterances retained.
- RMS is strictly positive for every utterance.
- 1,800 / 1,800 metric rows complete.
- 60 / 60 effect curves complete.
- 12 / 12 slope tests complete.
- No metric NaNs.

## Loudness provenance diagnostic

The source field relative_db is not equivalent to RMS dB:
- corr(relative_db, 20*log10(rms)) = -0.1087.

The public dataset card names relative_db but provides no computation semantics, so it cannot be
treated as a clearly defined absolute loudness measure.

## Relative-minus-Absolute slopes

Using source relative_db:

All features:
- Arousal: -0.0340, 95% CI [-0.0493, -0.0205].
- Dominance: -0.0234, [-0.0311, -0.0158].

Loudness only:
- Arousal: -0.0182, [-0.0227, -0.0136].
- Dominance: -0.0131, [-0.0182, -0.0078].

Using RMS dB = 20*log10(rms):

All features:
- Arousal: +0.0012, CI crosses zero.
- Dominance: +0.0137, 95% CI [0.0006, 0.0278].

Loudness only:
- Arousal: +0.0043, CI crosses zero.
- Dominance: +0.0242, 95% CI [0.0112, 0.0396].

## Registered Criteria

- Core-mechanism robustness with rms_db: not supported.
- Loudness-mechanism robustness with rms_db: not supported.
- Directional definition consistency: not supported.
- Rejection criterion is met: rms_db reverses Arousal and Dominance slope signs to positive in
  both all-feature and loudness-only analyses.

## Consequence

IEMOCAP results that rely on source relative_db must not be used as main evidence for an
absolute-versus-relative loudness claim. The IEMOCAP external validation should be restricted to
features with clear semantics, especially pitch and speaking rate, unless the original relative_db
generation procedure is recovered.

## Verdict

- Validity: valid.
- Decision: reject.
