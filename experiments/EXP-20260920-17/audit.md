# Experiment Audit — EXP-20260920-17

## Completeness

- Variance rows: 12 / 12.
- Between-speaker metric rows: 720 / 720.
- Between-speaker summary rows: 48 / 48.
- Within-speaker metric rows: 720 / 720.
- Within-speaker summary rows: 48 / 48.
- Within-speaker delta rows: 36 / 36.
- Metric NaNs: zero.

## VAD Variance Decomposition

Mean ICC-like between-speaker fraction:
- Valence: 0.213.
- Arousal: 0.418.
- Dominance: 0.346.

Thus Arousal and Dominance contain large stable between-speaker components.

## Speaker-Mean VAD Prediction

Using all three stable prosodic baseline components:
- Arousal speaker-mean CCC: 0.4825.
- Dominance speaker-mean CCC: 0.4412.
- Valence speaker-mean CCC: 0.0222.

Using baseline plus speaker marginal Relative:
- Arousal: 0.6688.
- Dominance: 0.6104.
- Valence: 0.0308.

## Within-Speaker Residual Prediction

After subtracting each speaker's full VAD mean from the target, Relative-minus-Absolute CCC becomes:

Arousal:
- Pitch: +0.0972.
- Loudness: +0.0883.
- Rate: +0.0008.
- All: +0.1306.

Dominance:
- Pitch: +0.0499.
- Loudness: +0.0761.
- Rate: +0.0008.
- All: +0.0921.

Valence remains weak.

Adding the stable speaker baseline back to Relative on residual targets changes CCC by only
approximately 0.0000-0.0005, confirming that baseline information is no longer useful once the
speaker-level target component has been removed.

## Registered Criteria

- Between-speaker affect structure: supported.
- Stable-baseline prediction of speaker-mean Arousal/Dominance: supported.
- Within-speaker Relative recovery: supported.

## Verdict

- Validity: valid.
- Decision: pass.
