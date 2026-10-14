# 058 risk forecasting and selective protection

Seed17 development exploration. Fixed 16384 TRAIN-answer pool, 128 candidates, 10 ordinary warmup epochs + 10 protected epochs.
Learners fit only early TRAIN trajectory/query groups. Late query/time holdout is risk prediction validation, not independent dataset confirmation.

| Predictor | Late AP (new drops) | Late recall at 10% (new drops) | Positive events |
|---|---:|---:|---:|
| boundary | 0.025214797075501165 | 0.6545454545454545 | 55 |
| pressure32 | 0.0884986938404818 | 0.8909090909090909 | 55 |
| pressure128 | 0.08805135936544906 | 0.9454545454545454 | 55 |
| lookahead | 0.10662057958604124 | 0.9454545454545454 | 55 |
| random | 0.0018087050936380607 | 0.05454545454545454 | 55 |
| logistic | 0.044670583595618256 | 0.9636363636363636 | 55 |
| tree | 0.15421950878919327 | 1.0 | 55 |

| Arm | New filtered MRR | Old VALID MRR | Old VALID drops | Old TRAIN drops | Old-head/new-gold ratio | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| new_only | 0.150805134 | 0.220174848 | 7 | 5418 | 1.000000 | reused |
| uniform | 0.149472384 | 0.220488438 | 4 | 4432 | 0.984249 | 1.5065612921025604 |
| boundary | 0.149664659 | 0.221005019 | 4 | 4038 | 0.966642 | 1.5078295718412846 |
| pressure32 | 0.147380412 | 0.221052110 | 6 | 3533 | 0.964944 | 2.2658669338561594 |
| pressure128 | 0.147353166 | 0.221027728 | 5 | 3542 | 0.956082 | 2.2566030640155077 |
| lookahead | 0.146606815 | 0.221053831 | 6 | 3533 | 0.965416 | 2.2433698209933937 |
| logistic | 0.147251843 | 0.221032905 | 5 | 3610 | 0.972408 | 2.4076790048275143 |
| tree | 0.147533306 | 0.221029459 | 5 | 3694 | 0.958061 | 2.5095005391631275 |

No global safety guarantee. Same query pool, same protected samples/strength rule; feature/routing costs are included in training.
Full-pool scans generate experiment labels; they are not features of the cheap predictors. Common trajectory costs and learner fitting are separately disclosed.
Exposed VALID used only after all training. No TEST, independent CPU audit, SHA closure, archive, or time limit.
