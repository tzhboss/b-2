# 059: candidate-directed simple protection

Seed17. Same ten-epoch warmup/Adam state and cached learners as058. Full128 vs top8 sampled candidates; no new label preparation.

| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old TRAIN drops | Old→new ratio | New→old ratio | New→new ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| new_only | 0.150805134 | 0.220174848 | 7 | 5418 | 1.000000 | 1.000000 | 1.000000 |
| uniform_all128 | 0.149472384 | 0.220488438 | 4 | 4432 | 0.984249 | 0.991402 | 1.002735 |
| uniform_top8 | 0.148512244 | 0.221011838 | 4 | 4376 | 0.979445 | 0.984125 | 1.003050 |
| current_all128 | 0.148078926 | 0.220550074 | 5 | 3656 | 0.968583 | 0.983075 | 0.996754 |
| current_top8 | 0.149288639 | 0.221012265 | 5 | 3681 | 0.969907 | 0.992753 | 1.000518 |
| new_event_all128 | 0.147606894 | 0.221043333 | 4 | 3692 | 0.958070 | 0.980881 | 0.998591 |
| new_event_top8 | 0.149516784 | 0.221047998 | 6 | 3625 | 0.967510 | 0.994555 | 1.006913 |

Ordinary and uniform/all128 reused, not independent repetitions. Exposed VALID, no TEST. No hard safety or novelty claim.
