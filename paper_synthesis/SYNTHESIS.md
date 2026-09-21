# Prosodic Reference Frames — Evidence Synthesis

## Central claim

The current experiments support a target-reference matching account rather than a universal
normalization advantage.

A speaker-level prosodic observation can be decomposed conceptually into speaker baseline,
within-speaker deviation, and other variation. A target can likewise contain a stable
between-speaker component plus a within-speaker component. The utility of Absolute, Relative, or
Hybrid prosody depends on which components the target rewards.

## Strongest evidence chain

1. Explicit controlled prosody: Relative pitch improves categorical emotion but removes
   speaker-trait information useful for gender classification (EXP-02).
2. Categorical heterogeneity: emotion-class and acoustic-attribute effects vary in sign
   (EXP-09/10); therefore Relative is better is not a valid universal claim.
3. Continuous VAD decomposition: MSP raw Arousal/Dominance contain large between-speaker
   components; after removing speaker target means, Relative becomes strongly better and baseline
   restoration becomes negligible (EXP-17).
4. External domain contrast: IEMOCAP has far less between-speaker VAD structure and correspondingly
   favors Relative much earlier (EXP-21-02).
5. Controlled intervention: continuously increasing between-speaker target strength makes
   Relative-minus-Absolute utility decrease in both corpora (EXP-21-03).
6. Model robustness: the intervention slope survives linear and quadratic Ridge (EXP-21-04).
7. Mechanism intervention: permuting speaker-center identity destroys raw MSP Hybrid utility but
   leaves within-speaker residual prediction nearly unchanged (EXP-21-06).
8. Deployment: unlabeled K-shot enrollment estimates useful speaker references; K around 20 is near
   a practical downstream plateau in MSP (EXP-20-19 / EXP-21-01 / EXP-21-07/08).
9. Semantically safe external replication: IEMOCAP pitch+rate alone preserve the coupling under
   both model families (EXP-21-10).

## Claims that should NOT be made

- Do not claim Relative prosody is universally superior.
- Do not claim Absolute prosody is universally superior for continuous affect.
- Do not use IEMOCAP relative_db as main loudness evidence; its semantics are unresolved and RMS-dB
  replacement reverses the relevant slope (EXP-21-09).
- Do not present the frozen-WavLM final-layer result as invariance; the representation retains highly
  decodable absolute, relative, and speaker-baseline pitch information.
- Do not use EXP-20 numerical results; that run is invalid due runtime provenance mismatch.
- Do not frame Valence as a strong success case for these low-dimensional prosodic features.

## Recommended paper-level statement

Prosodic normalization is an information transformation, not a universally beneficial nuisance
removal step. Speaker-relative representations are advantageous when the prediction target is
defined within speaker, whereas stable speaker baselines become useful when the target itself
contains between-speaker structure.

The controlled target intervention and center-identity permutation are the strongest mechanism
experiments supporting this statement.

## Additional trait-information boundary

A content-disjoint speaker-identity audit (EXP-20260921-11) shows that Relative all-prosody reduces
speaker-ID Macro-F1 consistently, but only modestly (roughly 0.018–0.035 absolute across ESD,
MEAD, and RAVDESS). Therefore the strong gender reversal should not be generalized to a claim that
speaker-relative normalization universally removes speaker identity. Explicit speaker baselines
still contain substantial identity information, and adding them back strongly increases
speaker-ID decodability.
