# TRAIN capacity follow-up draft; not a registered intervention

Complete030 and independent032 before selecting the next trained comparison.
Preserve027/028 failures, original gates and every common evaluation query.
034 measures implementation equivalence; it cannot justify a utility claim.

## Question and mechanism

The frozen shared graph already exempts prediction references at newTRAIN
query keys, but it retains all compiled TRUE anchors. Some affected queries
have ten old top-ranked answers which are all known old TRAIN facts. A top10
list with separated scores has insufficient capacity for ten such answers
plus a distinct new answer inserted above the old cutoff. Optimistic ranks
can retain more than ten answers when they tie at the cutoff; this argument
does not cover such ties, and a follow-up cannot optimize toward them.
This is a conflict between answer multiplicity and the protection
contract, not a solver tolerance or an unlabelled-negative inference.

An exploratory TRAIN-only diagnosis checks every old candidate score at these
queries, compares Torch and independently reduced NumPy FP64 trees exactly,
and rejects certificates whose top10 boundary is insufficiently separated
under a conservative FP32 rounding bound. Neither VALID nor TEST is read.
Raw outputs remain in `artifacts/rank-capacity-diagnostic/`; they are not
promoted scientific results. V1 lacks complete upstream identity replay;
V2 adds that check without overwriting V1's retained output.

## Conditional argument and explicit cost

If the k strictly separated old top-k answers are all in old TRAIN, a distinct
old gold outside TRAIN cannot have been a top-k success at that query. This
requires TRAIN/evaluation triple disjointness, complete old candidate scores,
frozen old E/R, a robust cutoff gap and unchanged unfiltered rank semantics.
No selection uses heldout query identities. Dataset split disjointness must
be checked independently before using this as a method certificate.

Exempting TRUE anchors at those affected keys changes preservation of known
TRAIN answers and may degrade old ranks outside top10. It does not preserve
all old knowledge. Any follow-up must report the number of exempted queries,
anchors and original TRAIN truths displaced, rather than hiding them behind
the unchanged old TEST Top10 loss metric. New candidates still compete in
every other compiled query, and those guards must remain active.

## Candidate comparison, conditional on complete032

Use the same common DistMult base, source NewEntityOnly checkpoints, seeds
17/29/43, 3753 new entity updates, dimension200, order, radius, solver margins,
iteration limits and complete common evaluator. No gradients update old E/R.
Do not increase the parameter budget or select coefficients from utility.

Compare the previously audited full and scoped frozen policies against:

1. Full P plus TRUE with only certified affected saturated-query exemptions.
2. Scoped P plus TRUE with only those same certified TRUE exemptions.

These are contract adaptations, not an improved solver or confirmed novelty.
Their difference isolates broad affected-query P exemption from TRAIN-certified
capacity exemption. Every untouched reference must reconstruct exactly from
024. Keep at least one unmodified control and record source-inclusive cost.
Reusing prior audited control artifacts is permissible only with exact source
identity binding; this must not be described as new independent retraining.

Keep the original utility gates: mean newMRR>=98% of source, mean oldTop10
loss<=75% of positive source loss and every seed nonworse, old-head/new-gold
MRR>=98% and Hits10>=95%. Require all three seeds and independent fixed VALID
replay. Report the complete TEST comparison as exposed development evidence.
Also require nonworse protected-reference coverage outside the certified keys.
Improved metrics alone do not certify the changed TRAIN protection contract.

## Evidence required before a run

Bind all source/checkpoint/scope identities; prove robust selection without
VALID/TEST access; independently check dataset split disjointness; reconstruct
all retained and exempted reference identities; test saturated, unsaturated,
tie and numerically ambiguous boundaries; and replay a small frozen-old update
fixture against the unmodified complete scoring/acceptance rule. Register the
final protocol/config only after these controls and complete032 disposition.

If data disjointness, fixed arithmetic or scope reconstruction fails, retain
the failure and do not grant the exemption. No optimistic tie manipulation,
weakened margin or threshold repair is allowed.

## Observed exploratory evidence on 2026-10-11

V2 completed in34.1 elapsed seconds on CPU and checked all upstream identities.
Three seeds certified321/324/282 queries covering3677/3447/2984 new TRAIN
facts. Respectively3207/3237/2819 initially eligible selected TRUE anchors
occur at those queries. Ten candidates at each key are known TRAIN truths;
the queries also have21809/21042/18297 additional old TRAIN truths outside
the original top10. The complete Torch/NumPy old-pool replay covered7024680
scores with zero difference. Six saturation/tie/error controls passed.
These figures describe TRAIN capacity, not method utility or causal TEST loss.

The separate `artifacts/rank-capacity-diagnostic/dataset-split-audit.json`
checked all TRAIN/VALID/TEST pair intersections for snapshots0 and1: all six
intersections were empty. That global integrity check read VALID/TEST without
model scoring, metrics or exemption selection. The V2 exemption certificates
were fixed from TRAIN before that separate check and remain unchanged.

Raw V1 and V2 summaries and arrays are retained. V1's original producer body
was preserved with the exact recorded implementation SHA. No model has been
updated, no032 gate has changed and no method result has been promoted here.
