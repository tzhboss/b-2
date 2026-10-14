# Protection mechanism search

| Arm | New MRR | Old VALID MRR | VALID drops | Old-new ratio | New-old ratio | New-new ratio | TRAIN drops |
|---|---:|---:|---:|---:|---:|---:|---:|
| new_only | 0.150805134 | 0.220174848 | 7 | 1.000000 | 1.000000 | 1.000000 | 5418 |
| uniform | 0.149472384 | 0.220488438 | 4 | 0.984249 | 0.991402 | 1.002735 | 4432 |
| historical_replay | 0.153509073 | 0.222396408 | 2 | 0.890150 | 1.042804 | 1.009809 | 2380 |
| benefit_ridge_static | 0.147639960 | 0.221030805 | 3 | 0.982179 | 0.976661 | 0.998040 | uncomputed |
| benefit_ridge_dynamic | 0.148226706 | 0.221093837 | 5 | 0.986674 | 0.980162 | 1.004926 | uncomputed |
| benefit_tree_static | 0.147206874 | 0.220549257 | 4 | 0.960629 | 0.977367 | 0.994609 | uncomputed |
| benefit_tree_dynamic | 0.147351995 | 0.221102571 | 3 | 0.970807 | 0.976636 | 0.995068 | uncomputed |
