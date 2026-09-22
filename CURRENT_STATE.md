# CURRENT_STATE.md

## Phase

Extended prosodic attributes, corrected speaker-cluster inference, and WavLM-level
target-reference intervention are complete.

## Strongest new findings

### Extended attributes
Target-reference coupling extends to F0 variability, pause ratio, and voiced ratio on MSP.

### Corrected inference
MSP slopes remain strongly negative under speaker-cluster bootstrap.
IEMOCAP hand-engineered Pitch+Rate slope means remain negative but CIs cross zero because there are
only ten speakers; pure within-speaker endpoints remain significant.

### Learned representation
Frozen WavLM layer 12 and 24 show significant negative Arousal/Dominance target-reference slopes
under speaker-cluster bootstrap on IEMOCAP, with all four pure within-speaker endpoints favoring
Relative embeddings.

## Paper interpretation

The main mechanism is now supported at two levels:
1. interpretable prosodic attributes;
2. learned SSL representation space.

The paper should no longer present WavLM only as a decodability appendix; EXP-20260921-16 is a
direct mechanism generalization and belongs in the main results or a strong secondary section.

## Next legal step

Update the manuscript, figures, and claim-evidence matrix to replace fold-level inference with
speaker-cluster inference and to add the WavLM target-reference intervention.
