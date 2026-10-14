# 062: limited old-row unfreezing

Seed17 exposed-VALID development screen; adaptations, not native TADG.

| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old-new ratio | New-old ratio | New-new ratio | Screen |
|---|---:|---:|---:|---:|---:|---:|---|
| standard_frozen | 0.150805134 | 0.220174848 | 7 | 1.000000 | 1.000000 | 1.000000 | reused |
| standard_frozen_absolute | 0.149472384 | 0.220488438 | 4 | 0.984249 | 0.991402 | 1.002735 | reused |
| slow_frozen | 0.153197729 | 0.219208792 | 9 | 0.852796 | 1.045270 | 1.030892 | False |
| slow_frozen_absolute | 0.153140916 | 0.220037912 | 8 | 0.849434 | 1.045244 | 1.032833 | False |
| slow_open64 | 0.136819974 | 0.194022949 | 40 | 0.882282 | 0.900085 | 1.036418 | False |
| slow_open64_absolute | 0.136580038 | 0.194924553 | 39 | 0.877289 | 0.898631 | 1.040314 | False |
| slow_open64_uniform_tar | 0.141512297 | 0.203456848 | 26 | 0.875134 | 0.941370 | 1.035549 | False |
| slow_open64_entropy_tar | 0.141512297 | 0.203458618 | 26 | 0.875134 | 0.941370 | 1.035549 | False |
| slow_open64_entropy_dggp | 0.150230727 | 0.212740859 | 15 | 0.865370 | 1.017564 | 1.032310 | False |
| standard_open64 | 0.112781230 | 0.170931502 | 89 | 0.999426 | 0.679248 | 0.977125 | False |

Only screened configurations receive full TRAIN evaluation; absent TRAIN metrics mean uncomputed.
New Adam moments/step370 retained; old opened rows fresh Adam step0. Relations and other old rows remain fixed.
All new runs share the 10-epoch prefix. Lower-rate comparisons use .0007 matched controls; standard pair uses .007.
No TEST, independent confirmation, time limit, audit, SHA gates, or archive.
