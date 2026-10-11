# Shared rank cache with explicit PyTorch binary reduction

The complete guard scope, grouped-reference equivalence, read-only proposals,
shared head/gold/competitor patch semantics and64 correctness cases remain
those in shared-rank-cache-protocol.md. The two failed Triton attempts remain
unchanged. This version replaces the unavailable JIT backend with ordinary
PyTorch operations. It needs no C compiler, Triton backend or package download.

Scoring: first multiply each shared head and relation coordinate in float64;
multiply each query coordinate by each candidate coordinate, pad the product
vector with zeros to the next power of two, then repeatedly add adjacent pairs
in separate PyTorch operations. Full-pool, subset and query-batch scores have
the same specified tree. Guard arithmetic differs from prior GEMM and CPU
scalar left-to-right sums; this is explicit, not an unchanged legacy formula
claim. Same strict-score rank definition, unchanged original common evaluation.
Independent scalar score tolerance1e-10 and exact rank equality remain required.

The64 randomized cases include multi-row head/gold/candidate changes, independent
all-member ranks, grouping equivalence, full-versus-subset/batch bitwise equality,
safe/unsafe commit paths, source immutability, stale patches, complete rescoring
after commits and explicit ties. A separate32-case independent audit uses a new
seed. The small-pool dimension7 checks padding; real dimension200/full11258
candidates require a scaling preflight before graph optimization.
