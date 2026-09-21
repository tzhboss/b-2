# CURRENT_STATE.md

## Phase

K=20 deployment target-reference intervention completed and audited.

## Main finding

MSP preserves strong target-reference coupling under K=20 unlabeled reference estimation.
IEMOCAP preserves the negative slope direction but confidence intervals cross zero, likely due
the combination of only ten speakers and noisy finite-K center estimates.

## Next legal step

On IEMOCAP only, reserve a fixed 100-utterance enrollment pool per speaker and use nested
K=[5,10,20,50,100] prefixes while keeping all downstream rows fixed. Measure whether K-shot
target-reference slopes converge toward the full-speaker oracle slopes.
