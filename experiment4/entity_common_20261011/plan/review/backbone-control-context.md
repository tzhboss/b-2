# Backbone-matched context for the common new-entity comparison

Derived from complete audited prior rows in common-comparison-v4. Raw arithmetic
and per-seed paired deltas are retained in
`artifacts/comparison-context/backbone-matched-controls/summary.json`.
No037 result, new model fit or utility gate was read or changed.

The common task/candidate pool/query sets match across methods, but their
backbones and parameter budgets differ. Native LKGE/IncDE/FastKGE/STARK/SAGE
adaptations use TransE; current QP variants use DistMult. Directly comparing
their final MRR or drop rates does not isolate the editing mechanism. Use
each backbone's audited NewEntityOnly control as additional context.

| Method | Backbone | New MRR change versus matching control | Old Top10 loss change versus matching control |
|---|---|---:|---:|
| LKGE | TransE | +2.24% | +1.16% |
| IncDE | TransE | -3.69% | +14.30% |
| FastKGE | TransE | +1.91% | +63.79% |
| STARK | TransE | +3.68% | +78.05% |
| SAGE | TransE | +6.47% | +69.63% |
| LearnedProposalQPPred | DistMult | -0.96% | -37.75% |
| SharedFrozenFull | DistMult | -0.96% | -37.75% |
| SharedFrozenScoped | DistMult | -0.64% | -29.15% |
| SharedJointScoped | DistMult | -6.08% | +46.93% |

Changes here mean ratio of three-seed means minus1, not mean seedwise ratio
and not percentage-point changes. Positive old-loss change is worse.
TransE control newMRR=.146656/old loss=8.3369%; DistMult control
newMRR=.141167/old loss=2.5915%. Their initial successful old-answer cohorts
have different sizes; normalized controls do not remove that distinction.

This descriptive comparison supports a learning/preservation tradeoff under
these common-task adaptations. It does not establish SOTA, significance,
author-original-protocol failure or superiority independent of backbone,
parameter budget or training cost. QP's lower overall old loss does not cure
its failed directional new-answer gate. Original032 dispositions remain
NO_GO.037/038 must complete their unchanged gates before any new utility claim.

The native methods preserve their original mechanism choices; these are not
all the same editor applied to the same scorer. A future contribution claiming
backbone independence needs a genuinely matched scorer implementation or a
separate controlled cross-backbone study. No such additional run is started
by this contextual calculation, and it must not overwrite current registries.
