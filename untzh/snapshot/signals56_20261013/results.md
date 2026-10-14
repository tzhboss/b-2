# Protection signal comparison 056

Seed17, same initial state and learning tape, 20 epochs. Old entity/relation rows fixed. New arms use 32 filtered-qualified TRAIN answers per step, complete new candidate pool, and initial 8-batch gradient RMS ratio 0.1. Reference arms are explicit MEOR adaptations, not author-native reproductions. Replay controls are reused 052/053 runs with their original sampling/strength; they are contextual tradeoff points, not identically calibrated signal controls.

Filtering uses TRAIN truths only. VALID is an exposed development set. No TEST evaluation or independent repetitions.

| Arm | New raw MRR | New filtered MRR | Old head → new gold filtered MRR ratio | New head → old gold ratio | Both new ratio | Old TRAIN filtered drops | Old VALID filtered drops | Old VALID filtered MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| new_only | 0.129662553 | 0.150805134 | 1.000000 | 1.000000 | 1.000000 | 5418 | 7 | 0.220174848 |
| replay_010 | 0.129157008 | 0.149749065 | 0.967516 | 0.996924 | 1.002597 | 4654 | 4 | 0.221220229 |
| replay_030 | 0.130697294 | 0.150994540 | 0.949041 | 1.012162 | 0.989879 | 3767 | 3 | 0.221781366 |
| replay_100 | 0.135178996 | 0.153509073 | 0.890150 | 1.042804 | 1.009809 | 2380 | 2 | 0.222396408 |
| absolute | 0.129847852 | 0.148288853 | 0.907002 | 0.995648 | 1.005842 | 1787 | 3 | 0.222079136 |
| relative | 0.106947737 | 0.124193210 | 0.709141 | 0.839669 | 0.882857 | 2077 | 31 | 0.213530750 |
| random | 0.108385697 | 0.126808158 | 0.794690 | 0.846766 | 0.871581 | 3280 | 35 | 0.213476390 |
| structural | 0.107446541 | 0.125024341 | 0.727967 | 0.842662 | 0.888420 | 2186 | 32 | 0.213452946 |

Timing records include actual shared reference preparation and initial calibration. Preparation computes both reference arms, so combined preparation is not a per-method deployment cost. Learning tape and controls were reused; their historical generation/training costs are not zero.

Full raw/filtered directional MRR, Hits1, Hits10, old MRR and fixed-eligibility drops are in evaluation/metrics.json.

No CPU audit, hash/freeze/closure gates, synthetic qualification loops, or archive stage were executed.
