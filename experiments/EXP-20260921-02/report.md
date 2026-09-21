# Experiment Report — EXP-20260921-02

## Main Result

The MSP mechanism partially replicates on IEMOCAP, with an informative domain difference.

IEMOCAP VAD contains little between-speaker target variance. Accordingly, raw Arousal and
Dominance do not show the strong Absolute/Hybrid preference observed in MSP; Arousal instead
strongly favors Relative prosody.

Within-speaker residual prediction does replicate cleanly: Relative prosody strongly outperforms
Absolute for Arousal and Dominance, while adding stable speaker baseline information back produces
almost no additional benefit.

## Cross-corpus implication

MSP and IEMOCAP together suggest that the relevant variable is not corpus identity itself but
the reference frame of the target:
- high between-speaker target structure -> stable acoustic baseline can be useful;
- low between-speaker target structure -> Relative can already be preferable on raw targets;
- explicitly within-speaker targets -> Relative is consistently preferable.

## Decision

mixed.
