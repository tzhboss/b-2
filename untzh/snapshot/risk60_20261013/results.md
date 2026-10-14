# 060: direction coordination after risk identification

Single seed17 development experiment. Same warmup, query/candidate cache, loss, strength cap and classifier. One-direction projection is not a ranking safety guarantee.

| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old→new ratio | New→old ratio | New→new ratio | Quality screen |
|---|---:|---:|---:|---:|---:|---:|---|
| new_only | 0.150805134 | 0.220174848 | 7 | 1.000000 | 1.000000 | 1.000000 | reused |
| uniform | 0.149472384 | 0.220488438 | 4 | 0.984249 | 0.991402 | 1.002735 | reused |
| risk_current | 0.148078926 | 0.220550074 | 5 | 0.968583 | 0.983075 | 0.996754 | reused |
| uniform_euclidean | 0.149476002 | 0.220488493 | 4 | 0.984250 | 0.991433 | 1.002734 | True |
| risk_euclidean | 0.147917083 | 0.221047333 | 5 | 0.953236 | 0.984864 | 0.993865 | False |
| uniform_adammetric | 0.149476000 | 0.220488441 | 4 | 0.984250 | 0.991432 | 1.002734 | True |
| risk_adammetric | 0.147917904 | 0.221047333 | 5 | 0.953238 | 0.984870 | 0.993864 | False |

All candidates receive VALID screening after training. Only quality-screen candidates receive expensive full TRAIN ranking. Failed candidates have TRAIN metrics uncomputed, not zero.
Historical controls reused; no independent repetitions, TEST, audit/closure/archive, or fixed timeout.
