# Exploratory original control costs at fixed capacity query keys

Raw complete evidence:
`artifacts/rank-capacity-diagnostic/original-control-costs/summary.json`.
This reads only previously audited source/full/scoped endpoints and the
fixed TRAIN-certified saturated keys. No037 outputs or VALID/TEST metrics
were read. No038 gate, source, coefficient or candidate selection changed.

| Seed | Fixed certified keys | Original top10 TRAIN truths | Source SGD lost | Full/scoped frozen lost |
|---|---:|---:|---:|---:|
| 17 | 321 | 3210 | 113 | 2/2 |
| 29 | 324 | 3240 | 32 | 3/3 |
| 43 | 282 | 2820 | 21 | 0/0 |

Source SGD endpoint MRR on these truths is.290369/.291800/.292646;
full/scoped frozen endpoint MRR is.292870/.292871/.292897. No excess
cutoff ties occur in any of these nine endpoints. Full-pool NumPy ranks were
computed at each exact certified query key, then gathered for the ten old
TRAIN answers. These are contextual baseline costs, not new trained outcomes.

The registered038 additional gate compares exempted TRAIN costs to useful
unprotected source SGD. Passing it would not prove nonworse costs relative
to the stronger original QP/full/scoped controls. Any improved new-answer
retrieval must therefore report its original-control TRAIN truth tradeoff
separately. Do not call a changed preservation contract unconditional
dominance over QP or an all-old-knowledge guarantee.

The original diagnostic was stopped by the agent after accidentally using
about27 of32 CPU cores for tiny repeated BLAS calls. Its interruption and
partial observations are retained under
`artifacts/rank-capacity-diagnostic/original-control-costs-threading-stop/`.
The identical model/certificate calculation reran with two BLAS threads and
completed all seeds. Main037/034 jobs were neither stopped nor modified.
