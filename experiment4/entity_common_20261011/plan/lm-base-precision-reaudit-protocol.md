# Explicit CPU precision adjudication of an unchanged clean foundation

LMBASE004 seed17 float32 CPU replay failed one strict rank at VALID fixture row60 (rank241 vs GPU240). LM-NUMERICS015 diagnosed candidate3298 vs gold2238 margins -4.76837158203125e-7 on GPU and +1.430511474609375e-6 on CPU. CPU batches8/16 both differ; GPU reload is bitwise exact; native CPU float64 all64 fixture strict ranks match GPU. No oldTRAIN/newTRAIN/TEST experiment or scientific parameter changed.

New LMBASE-REAUDIT016 explicitly preregisters native CPU float64 batch8 and GPU float32 batch16 replay for all64 queries and7505 candidates. Require the existing1e-4 score tolerance and exact ranks for CPU float64, and zero score error plus exact ranks for GPU float32 restoration. This changes CPU arithmetic precision, not score tolerance or rank equality. It is a new precision contract informed by the disclosed015 diagnostic, not a retroactive PASS under original004.

Also verify all original foundation input hashes, exact checkpoint hash, new embedding-row initialization hash and first-best oldVALID checkpoint selection. Original004 checkpoint, fixture, audit and failure remain immutable. Hard-link source checkpoint/terminal/training/fixture into a separate versioned base directory; write only the new audit there. A failure is retained. A pass qualifies only this exact unchanged base for a separate seed17 KGEditor continuation. Native meta/online scientific options and original restore/final audits remain required. Any later failure remains a failure; this adjudication is not advance permission to relax them.

Report CPU precision and retained original failure in the common comparison provenance. No claim of independent architecture or optimizer reproduction.
