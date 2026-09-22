# Experiment Audit — EXP-20260921-18

## Runtime integrity

- Completed with exit code 0 on 2026-09-22.
- Immutable runtime bundle script/config hashes match the registered source hashes.
- Runtime commit: d5ae82ee7883ec2ebfba92bc9668ab8bf69efe7f.
- Dataset inventory: 10,039 downstream rows, 10 speakers, minimum 859 rows per speaker.
- Enrollment pool: 50 unlabeled utterances per speaker.
- Enrollment seeds: 20260920, 20260921, 20260922.
- Speaker is the bootstrap unit; enrollment seeds are nuisance realizations averaged within bootstrap replicates.

## Registered acceptance checks

K=20 slope means:
- layer 12 Arousal: -0.028234, 95% CI [-0.040868, -0.016109].
- layer 12 Dominance: -0.001325, [-0.026726, 0.019739].
- layer 24 Arousal: -0.044648, [-0.078445, -0.012209].
- layer 24 Dominance: -0.072859, [-0.125115, -0.021726].

All four K=20 slope means are negative: supported.

K=20 lambda=0 effects:
- layer 12 Arousal: +0.016739.
- layer 12 Dominance: -0.000272.
- layer 24 Arousal: +0.033455.
- layer 24 Dominance: +0.019134.

Three of four are positive: supported.

K=50 oracle comparison:
- layer 12 Arousal slope gap: 0.015841.
- layer 12 Dominance slope gap: 0.005619.
- layer 24 Arousal slope gap: 0.026153.
- layer 24 Dominance slope gap: 0.043257.

All four K=50 slope signs match oracle and three of four slope gaps are <=0.03: supported.

## Caveat

The K curve is not monotonic. In particular, layer-24 Dominance has a positive slope at K=5 (+0.069498) before returning negative at K=10/20/50. The evidence supports practical recovery with moderate enrollment, not universal stability at tiny K.

## Verdict

- Validity: valid.
- Decision: pass.
