# Experiment Audit — EXP-20260921-12

## Runtime integrity

- Source/runtime script SHA256 match.
- Source/runtime config SHA256 match.
- Immutable /tmp execution used.
- MSP: 197,020 rows, 1,915 speakers.
- IEMOCAP: 10,039 rows, 10 speakers.
- 2,700 / 2,700 metric rows complete.
- 60 / 60 effect-curve rows complete.
- 24 / 24 slope rows complete.
- No metric NaNs.

## Semantically matched Pitch+Rate slopes

Relative-minus-Absolute CCC slope versus target between-speaker strength lambda.

Linear Ridge:
- MSP Arousal: -0.3181, 95% CI [-0.3264, -0.3082].
- MSP Dominance: -0.1821, [-0.1878, -0.1750].
- IEMOCAP Arousal: -0.02245, [-0.03772, -0.00838].
- IEMOCAP Dominance: -0.01089, [-0.01684, -0.00348].

Quadratic Ridge:
- MSP Arousal: -0.3207, 95% CI [-0.3293, -0.3103].
- MSP Dominance: -0.1846, [-0.1906, -0.1771].
- IEMOCAP Arousal: -0.02646, [-0.04310, -0.01134].
- IEMOCAP Dominance: -0.01752, [-0.02440, -0.00994].

All eight confirmatory slopes are significantly negative.

## Pure within-speaker endpoint lambda=0

Relative-minus-Absolute CCC:
- MSP Linear: Arousal +0.1389, Dominance +0.0727.
- MSP Quadratic: Arousal +0.1389, Dominance +0.0746.
- IEMOCAP Linear: Arousal +0.1363, Dominance +0.0458.
- IEMOCAP Quadratic: Arousal +0.1551, Dominance +0.0486.

All eight endpoint effects are positive.

## Registered Criteria

- Cross-corpus mechanism replication: supported.
- Within-speaker endpoint replication: supported.
- Strong semantic-match robustness: supported.
- Rejection criterion: not met.

## Verdict

- Validity: valid.
- Decision: pass.
- This is the cleanest confirmatory cross-corpus evidence for the main target-reference matching claim.
