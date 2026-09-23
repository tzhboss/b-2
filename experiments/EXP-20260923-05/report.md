# Experiment Report — EXP-20260923-05

## Audit repair

This experiment repairs two problems in EXP-20260922-02:
1. enrollment seeds are now derived with SHA-256 and genuinely select different reserves;
2. uncertainty is computed by paired speaker-cluster bootstrap over 884 speakers rather than treating
   seed×fold cells as independent observational units.

The three K=50 enrollment reserves have three distinct SHA-256 hashes.

## Corrected result

Hybrid minus Absolute under speaker-cluster inference:

- K=20 Arousal: **+0.04205**, 95% CI [+0.03338,+0.05052].
- K=20 Dominance: **+0.03559**, [+0.02830,+0.04307].
- K=50 Arousal: **+0.04933**, [+0.03958,+0.05875].
- K=50 Dominance: **+0.04175**, [+0.03342,+0.05004].

K=50 remains close to the marginal-oracle Hybrid:
- Arousal K50 minus oracle: **-0.00284**, [-0.00451,-0.00119].
- Dominance: **-0.00306**, [-0.00477,-0.00141].

The K50−K20 gain is:
- Arousal **+0.00729**, [+0.00461,+0.00994].
- Dominance **+0.00616**, [+0.00350,+0.00871].

Thus both point estimates and the upper confidence limits remain below the preregistered 0.01
plateau threshold.

## Interpretation

The practical-enrollment conclusion survives corrected randomization and corrected inference.
Roughly 20 unlabeled utterances already recover a substantial speaker-reference benefit; 50
utterances recover most of the oracle-center gain under this six-feature Ridge setup.

Seeds are repeated modeling conditions only. The inferential unit is the speaker.

## Decision

pass.
