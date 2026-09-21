# CURRENT_STATE.md

## Phase

Fixed-pool IEMOCAP K-shot reference-quality curve completed and audited.

## Main finding

Physical speaker-center estimation improves strongly with K, but deployment target-reference slope
recovery is non-monotonic. K=50 is closer to the oracle mechanism than K=100 for Arousal and
Dominance on this ten-speaker corpus.

## Next legal step

Audit the semantic/provenance meaning of IEMOCAP acoustic fields, especially relative_db, before
using the IEMOCAP attribute-specific results as paper evidence.
