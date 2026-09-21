# CURRENT_STATE.md

## Phase

Nested K-shot VAD sample-efficiency experiment completed; fixed-pool correction required.

## Main valid result

Acoustic-center estimation improves monotonically through K=50. K=20 already reaches near-oracle
Arousal/Dominance Hybrid CCC, but the apparent K20-to-K50 Arousal drop is confounded because larger
K removes more downstream rows.

## Next legal step

Reserve the same 50 label-free enrollment utterances for every speaker and every K, keep all
downstream train/test rows fixed, and estimate the acoustic center from nested prefixes
K=[1,2,5,10,20,50].
