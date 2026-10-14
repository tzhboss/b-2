# Protection mechanism search

| Arm | New MRR | Old VALID MRR | VALID drops | Old-new ratio | New-old ratio | New-new ratio | TRAIN drops |
|---|---:|---:|---:|---:|---:|---:|---:|
| new_only | 0.150805134 | 0.220174848 | 7 | 1.000000 | 1.000000 | 1.000000 | 5418 |
| uniform | 0.149472384 | 0.220488438 | 4 | 0.984249 | 0.991402 | 1.002735 | 4432 |
| historical_replay | 0.153509073 | 0.222396408 | 2 | 0.890150 | 1.042804 | 1.009809 | 2380 |
| host_mass_old | 0.147312202 | 0.221653748 | 4 | 0.942306 | 0.981214 | 1.000122 | uncomputed |
| host_mass_gold | 0.148815959 | 0.221659964 | 4 | 0.945140 | 0.993237 | 1.002441 | uncomputed |
| host_mass_guard | 0.147062131 | 0.221711230 | 4 | 0.940326 | 0.979445 | 1.000326 | uncomputed |
| host_mass_sample | 0.149052430 | 0.220455216 | 4 | 0.997088 | 0.985740 | 0.999158 | 4354 |
| contrastive_none | 0.177636074 | 0.175947573 | 99 | 1.955752 | 1.000728 | 1.507231 | uncomputed |
| contrastive_absolute | 0.174425534 | 0.197320649 | 68 | 1.709605 | 1.026467 | 1.436287 | uncomputed |
| contrastive_semantic | 0.175375500 | 0.196813253 | 70 | 1.720515 | 1.032532 | 1.435674 | uncomputed |
| contrastive_mass_old | 0.170042247 | 0.206311222 | 51 | 1.544942 | 1.028874 | 1.343464 | uncomputed |
| fullsoftmax_none | 0.161709073 | 0.070392887 | 280 | 3.170107 | 0.647366 | 1.385672 | uncomputed |
| fullsoftmax_absolute | 0.162794113 | 0.114325736 | 191 | 3.041658 | 0.677347 | 1.423533 | uncomputed |
| fullsoftmax_semantic | 0.164269363 | 0.113240579 | 191 | 3.044341 | 0.683679 | 1.485327 | uncomputed |
| fullsoftmax_mass_old | 0.158749542 | 0.098170125 | 221 | 2.979634 | 0.662604 | 1.337754 | uncomputed |
