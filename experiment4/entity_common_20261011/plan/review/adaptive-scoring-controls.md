# Optional adaptive query batching: engineering qualification

The active 034/037 scorers, solvers, configurations, model trajectories and
scientific utility gates remain unchanged. This optional scorer retains the
original FP64 multiplication, padding, column tiles and pairwise reduction
tree. Query batches grow for narrow column pools while the peak product-cell
bound remains 64*256. Full-width pools keep the original query batching.

The original CPU shape output remains at
`artifacts/adaptive-rank-scoring/cpu-shapes/summary.json`. Its equality checks
used numeric equality, which alone cannot distinguish signed-zero bit patterns.
The strengthened checker therefore adds signed-zero fixtures and checks all
64-bit representations against both the original Torch scorer and an
independent NumPy tree. Its separate CPU output covers 14 shapes and 135514
score cells, all passing, under
`artifacts/adaptive-rank-scoring/cpu-bitwise-shapes/summary.json`.

The real CPU check uses only the original audited seed17 full protection
scope (68270 TRAIN query groups, 235882 reference members) and initial model
rows 7505/7506. No model update or VALID/TEST scoring occurs. All 204810 real
scores match original Torch and independent NumPy bit patterns. Five alternating
timed repetitions after warmup gave these per-call medians with two CPU threads:

| Initial candidate rows | Original seconds | Adaptive seconds | Speed ratio |
|---|---:|---:|---:|
| One | 0.20185950 | 0.05731590 | 3.52 |
| Two | 0.23836294 | 0.10526472 | 2.26 |

These are narrow-score-call measurements from one concurrent CPU session,
not graph trajectory equivalence, end-to-end speedup or scientific benefit.
Inputs and source identities are bound in the raw summary at
`artifacts/adaptive-rank-scoring/cpu-real-train/summary.json`; lightweight copies
are under `results/engineering-adaptive-scoring/`.

CUDA qualification subsequently passed on GPU6 after the034 seed29 joint
worker terminated and its audit completed.034 had no remaining queued jobs,
its other workers occupied GPUs4/5,037 occupied GPUs1/2, and GPU6 had zero
utilization and no compute-process claim at inspection. Both diagnostic
processes terminated successfully and did not update any model.

CUDA shape controls check 135514 cells with the same14 configurations, all
bitwise equal to the original Torch scorer and independent NumPy tree. The
real TRAIN check adds204810 cells, also bitwise equal. Five alternating timed
repetitions after warmup on an RTX3090 yielded:

| Initial candidate rows | Original seconds | Adaptive seconds | Speed ratio |
|---|---:|---:|---:|
| One | 0.31937757 | 0.00161944 | 197.21 |
| Two | 0.32199811 | 0.00283105 | 113.74 |

Raw CUDA summaries are under `artifacts/adaptive-rank-scoring/` with lightweight
copies under `results/engineering-adaptive-scoring/`. HAMI emitted existing
memory-accounting and host-PID warnings, but both processes returned exit0
and all requested score-bit assertions passed. Its memory accounting is not
used to claim whole-server resource consumption or unconstrained portability.

Narrow-score-call speed ratios do not predict an equal graph/runtime speedup.
This is one device/session and fixed initial rows; complete shared graph
trajectory equivalence remains untested. A source-bound trajectory check
would still be required before future integration. No current registered
scorer, source/config or running037/034 algorithm was modified.
