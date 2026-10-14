# Protection mechanism search

| Arm | New MRR | Old VALID MRR | VALID drops | Old-new ratio | New-old ratio | New-new ratio | TRAIN drops |
|---|---:|---:|---:|---:|---:|---:|---:|
| new_only | 0.150805134 | 0.220174848 | 7 | 1.000000 | 1.000000 | 1.000000 | 5418 |
| uniform | 0.149472384 | 0.220488438 | 4 | 0.984249 | 0.991402 | 1.002735 | 4432 |
| historical_replay | 0.153509073 | 0.222396408 | 2 | 0.890150 | 1.042804 | 1.009809 | 2380 |
| shared_none | 0.150805131 | 0.220174848 | 7 | 1.000000 | 1.000000 | 1.000000 | uncomputed |
| shared_absolute | 0.149472381 | 0.220488438 | 4 | 0.984249 | 0.991402 | 1.002734 | 4432 |
| split_original_none_1 | 0.146817652 | 0.218714339 | 11 | 1.004309 | 0.964043 | 1.013845 | uncomputed |
| split_original_absolute_1 | 0.147283237 | 0.219736655 | 8 | 0.989991 | 0.970604 | 1.014891 | uncomputed |
| split_original_absolute_3 | 0.147120700 | 0.220434593 | 6 | 0.945491 | 0.978652 | 1.003747 | uncomputed |
| split_original_mass_sample_1 | 0.147075901 | 0.219535492 | 8 | 0.987011 | 0.969501 | 1.013849 | uncomputed |
| split_mixed_none_1 | 0.154081299 | 0.215081043 | 19 | 1.079123 | 1.003758 | 1.099130 | uncomputed |
| split_mixed_absolute_1 | 0.153866699 | 0.217127468 | 13 | 1.040815 | 1.010533 | 1.084308 | uncomputed |
| split_mixed_absolute_3 | 0.152554465 | 0.219339246 | 7 | 0.996092 | 1.012676 | 1.031708 | uncomputed |
| split_mixed_mass_sample_1 | 0.154150477 | 0.216376470 | 16 | 1.061437 | 1.008838 | 1.086609 | uncomputed |
| split_contrastive_none_1 | 0.169218186 | 0.175033728 | 108 | 1.765732 | 0.985669 | 1.283953 | uncomputed |
| split_contrastive_absolute_3 | 0.163127533 | 0.208502019 | 35 | 1.401821 | 1.022084 | 1.072898 | uncomputed |
| split_contrastive_absolute_10 | 0.154141597 | 0.219403291 | 7 | 1.032490 | 1.023029 | 0.991048 | uncomputed |
| split_contrastive_mass_sample_3 | 0.169336527 | 0.198915182 | 66 | 1.700039 | 1.010582 | 1.159041 | uncomputed |
