# Route A method / SOTA evidence matrix — 2026-09-24

## Scope
Standard absolute SER is the real task. Relative / Between / Within are diagnostic decompositions only.

## Same-protocol evidence (primary)
All primary comparisons below use IEMOCAP strict 5-session LOSO, nested session-LOSO selection where applicable, and paired speaker-cluster inference.

### Baseline vs decomposition-guided method
From EXP-20260924-11 / EXP-20260924-13:
- Arousal component-balanced maximin vs nested best single:
  - Overall +0.00960 (default Ridge), +0.00905 (LSQR)
  - Between +0.00996, +0.00594
  - Within +0.00967, +0.00934
- Dominance component-balanced maximin vs nested best single:
  - Overall +0.02593, +0.02507
  - Between +0.13630, +0.13228
  - Within +0.01948, +0.01885
- Dominance deltas are significant on all three components under both solvers.
- Arousal Overall and Within are significant; Between remains positive but uncertain.

### Against strong generic ensemble
From EXP-20260924-14:
- Versus inner-CV convex SSL ensemble, maximin is statistically indistinguishable on all three components for both targets.
- Overall point deltas are -0.00040 (Arousal) and -0.00240 (Dominance).
Interpretation: the decomposition-guided method is competitive with a strong generic nested ensemble, not superior to it.

### Against published personalization/calibration baseline
From EXP-20260924-14:
- maximin vs PLDC mu+sigma:
  - Arousal Overall +0.04026, 95% CI [0.01647, 0.06772]
  - Arousal Within +0.02443, 95% CI [0.00386, 0.04587]
  - Dominance Overall +0.09450, 95% CI [0.00515, 0.17927]
- Between deltas are positive but have wide uncertainty because IEMOCAP has only 10 speakers.

## Public-protocol / external SOTA context
EXP-20260924-12 reproduces a fixed IEMOCAP S1-3 train / S4 validation / S5 test comparison protocol.

Our results on that split:
- Arousal:
  - best single 0.7152
  - equal SSL ensemble 0.7309
  - validation convex ensemble 0.7382
  - component-balanced maximin 0.7218
- Dominance:
  - best single 0.4992
  - equal SSL ensemble 0.5427
  - validation convex ensemble 0.5484
  - component-balanced maximin 0.5490

PCM (Interspeech 2025) published reference:
- Arousal 0.744
- Dominance 0.557

Gap to published PCM:
- strongest tested Arousal: 0.7382, gap -0.0058
- strongest tested Dominance: 0.5490, gap -0.0080

Important: the PCM line is literature context, not the same trained architecture. Do not label our model as beating or matching SOTA unless the architecture/training and exact protocol are directly comparable.

## Recommended paper claim
The evidence supports:
1. Overall CCC conflates distinct Between-speaker calibration and Within-speaker affect-tracking behavior.
2. Published personalization/calibration and normalization methods can move these components in different or opposite directions.
3. The decomposition can guide a competitive intervention: component-balanced maximin improves both components relative to a nested best-single baseline and remains stable under solver sensitivity.
4. Under identical nested session-LOSO, the proposed intervention substantially outperforms the PLDC mu+sigma baseline, while remaining statistically indistinguishable from a strong generic inner-CV convex ensemble.
5. Under the PCM public split, strong frozen-SSL ensembles are close to the published PCM Arousal/Dominance CCC, but this is external context rather than an apples-to-apples SOTA claim.

## Remaining highest-value gaps
- Cross-dataset replication of the component-balanced method would strengthen generality, but current MSP assets lack the same frozen SSL panel and would require new feature extraction.
- IEMOCAP Between-speaker uncertainty is intrinsically wide with only 10 speakers; avoid overselling nonsignificant Between deltas.
- If compute budget permits, MSP frozen SSL extraction is the next major experiment; otherwise the present IEMOCAP + MSP mechanism evidence is already sufficient for a conservative method claim.
