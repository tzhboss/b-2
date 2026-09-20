# Experiment Report — EXP-20260920-02

## Registered Hypothesis

Inherited unchanged from EXP-20260920-01 after results exposure: speaker-relative prosody
will improve emotion classification relative to absolute prosody, whereas absolute prosody
will improve gender classification; the magnitude of this trade-off will vary across pitch,
loudness, and rate, with pitch expected to show the strongest trait-versus-state contrast.
Hybrid features may recover complementary information and remain competitive with the better
single reference frame. EXP-20260920-02 is a metric/fold correction, not a fresh preregistration.

## Results

The corrected run completed 864 matched fold-level evaluations across ESD-English, MEAD-part0,
and RAVDESS-speech. All comparisons use three seeds and four full-label within-speaker folds.

For pitch, Relative minus Absolute macro-F1 was:
- ESD emotion: +0.0908 [0.0870, 0.0947]; gender: -0.1330 [-0.1390, -0.1265].
- MEAD emotion: +0.0904 [0.0876, 0.0933]; gender: -0.3790 [-0.3817, -0.3764].
- RAVDESS emotion: +0.0586 [0.0417, 0.0724]; gender: -0.2299 [-0.2427, -0.2158].

For all three attributes jointly, Relative minus Absolute macro-F1 was:
- ESD emotion: +0.0518 [0.0485, 0.0550]; gender: -0.1718 [-0.1785, -0.1650].
- MEAD emotion: +0.0591 [0.0567, 0.0615]; gender: -0.3861 [-0.3897, -0.3828].
- RAVDESS emotion: +0.0149 [0.0002, 0.0296]; gender: -0.2183 [-0.2382, -0.1972].

Attribute effects are not uniform. For emotion, pitch consistently benefits from speaker-relative
representation, while loudness is positive only on MEAD and rate is positive on RAVDESS but near
zero on ESD. For gender, pitch consistently favors absolute representation, whereas loudness is
near zero on MEAD and favors relative representation on ESD and RAVDESS.

Hybrid representations often recover complementary information. In the all-attribute gender
probe, hybrid macro-F1 reaches 1.0000 on ESD, 0.99997 on MEAD, and 1.0000 on RAVDESS.

## Observation

The direction and magnitude of the Absolute-versus-Relative effect vary jointly with task and
attribute. Pitch gives the cleanest cross-dataset reversal: Relative is better for emotion in all
three datasets, while Absolute is better for gender in all three. Loudness and rate do not follow
one universal direction. Hybrid features can substantially outperform either single reference
frame, especially for gender.

## Supported Claim

Under this reference-available, known-speaker, controlled-corpus probe, there is no single
universally best prosodic reference frame. The utility of speaker-relative versus absolute
prosody is task- and attribute-dependent. Pitch provides replicated evidence across all three
datasets that relative representation is more informative for emotion classification while
absolute representation retains substantially more gender-related information.

## Unsupported Stronger Claim

This experiment does not establish causality, human perceptual mechanisms, performance of a
speech foundation model, cross-corpus transfer, unseen-speaker generalization, or superiority of
speaker-relative prosody for emotion recognition in general. It also does not show that gender
should be inferred from speech in deployed systems.

## Post-experiment Interpretation

A plausible information-decomposition interpretation is that speaker-relative transforms suppress
stable speaker baselines while preserving within-speaker state deviations. This interpretation is
especially consistent with pitch. The perfect or near-perfect hybrid gender probes should be
treated carefully: combining an absolute value with its speaker-relative deviation can expose the
speaker reference/baseline algebraically, making trait information unusually easy to recover.
This is expected in the registered known-speaker/reference-available setting but must not be
confused with unseen-speaker performance.

## Decision

pass. The preregistered task-dependent criterion is met by pitch in all three datasets, exceeding
the requirement of at least two datasets with effect magnitude >= 0.01 and paired-bootstrap 95%
confidence intervals excluding zero in opposite directions for emotion versus gender. Attribute
dependence is also supported because loudness and rate show materially different directions or
resolved winners from pitch in multiple datasets/tasks.

## Next Step

Extend the same frozen protocol to uncontrolled MSP/MELD where possible, then register a separate
unseen-speaker/reference-estimation experiment. A later model-level experiment can test whether
the same reference-frame interaction persists when explicit prosody is fused with a frozen speech
encoder such as WavLM.
