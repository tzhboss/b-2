# Experiment Audit — EXP-20260921-07

## Runtime integrity

- Source/runtime script SHA256 match.
- Source/runtime config SHA256 match.
- Immutable /tmp execution used.
- MSP: 1,522 speakers with >20 eligible rows.
- IEMOCAP: all 10 speakers retained.
- 1,350 / 1,350 metric rows complete.
- 30 / 30 effect curves complete.
- 12 / 12 slope tests complete.
- No metric NaNs.

## K-shot Relative-minus-Absolute slopes versus lambda

MSP:
- Arousal: -0.4584, 95% CI [-0.4654, -0.4501].
- Dominance: -0.3661, [-0.3740, -0.3574].

IEMOCAP:
- Arousal: -0.0105, [-0.0307, +0.0081].
- Dominance: -0.0087, [-0.0202, +0.0038].

Both IEMOCAP slopes preserve the expected negative direction but are unresolved.

## Lambda=0 endpoints

K-shot Relative minus Absolute:
- MSP Arousal: +0.1782.
- MSP Dominance: +0.1337.
- IEMOCAP Arousal: +0.0792.
- IEMOCAP Dominance: +0.0162.

Endpoint matching is supported in all four Arousal/Dominance corpus-target cells.

## MSP K-shot center utility slope

K-shot Hybrid minus K-shot Relative:
- Arousal: +0.3008, 95% CI [0.2905, 0.3101].
- Dominance: +0.2401, [0.2311, 0.2482].

The preregistered MSP center-utility criterion is supported.

## Verdict

- Validity: valid.
- Decision: mixed.
- MSP deployment target-reference coupling: supported.
- IEMOCAP deployment direction: preserved but statistically unresolved at K=20.
- Endpoint matching: supported.
