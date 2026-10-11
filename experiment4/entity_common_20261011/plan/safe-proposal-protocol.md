# Learned proposal with rank protection: optimization candidate

## Scope and motivation

Known Euclidean constrained projection and existing SVD directions, not a confirmed novel algorithm. The current raw/bounded sequential promotion method can destabilize shared vectors; its uncertified fallback has no rank guarantee. This candidate changes the proposal source to the audited NewEntityOnly/DistMult200 final shared model. It retains all new TRAIN learning already performed by that control, then applies one deterministic sweep over all3753 new entity vectors. This is a development method candidate, not an original Rank-paper reproduction or a heldout superiority claim.

No real-graph optimization run starts before common comparison batches007,008 and corrected audit010 are terminal. Implementation and synthetic checks may proceed while they run. Existing code/config/results remain immutable. The proposed complete comparison has18 jobs: LearnedProposal, LearnedProposalRank, LearnedProposalQPTrue, LearnedProposalQPPred, LearnedProposalRankQPTrue and LearnedProposalRankQPPred ×seeds17/29/43. These are added as qualified rows to the same common table, not a separate easier task.

## Fixed data and proposals

Same FB_CKGE snapshot0→1,11258 candidates, old/new TEST74428/12406 raw strict-greater queries. Use the same DistMult200 base and random expansion values (float32 values, float64 arithmetic). Verify source NewEntityOnly checkpoint/audit/hash, exact oldE/R immutability, original learning input hashes and TRAIN-only selection receipt. Reused SGD phase costs remain visible; postprocessing time cannot be presented as total learning cost.

Old TRAIN guard source is existing audited per-seed guard cache, with selected TRUE old7505 Top10 facts capped5000/relation and68334 unique prediction queries. The candidate protects only the initially retained full11258-pool members of these registered sets. It does not claim to undo initialization losses, protect all TEST, or retain every numerical rank. Old representations/relations remain bitwise fixed. Strong source SGD proposals and real shared head/tail evaluation prevent treating an empty or no-update method as a useful result.

## Fixed mechanism

Entity order: uniform permutation of3753 new IDs, seed+271, exactly one update per entity. Proposal is source trained vector minus current vector. Rank arm removes the topceil(.125*rank) right-singular directions of the all-relation TRUE query matrix before projecting, without forced Top10 promotion or re-amplifying the residual. This is an explicit all-relation learned-proposal adaptation, distinct from the original relation-specific analytic Rank formula. Derive the matrix/basis from old TRAIN only and record its hash/eigenvalues.

For each current protected frontier with9 strictly higher competitors, a candidate not already above the cutoff has affine constraint `Q delta <= cutoff-Q old`. Slack is nonnegative. Subtract only `min(1e-8, slack/2)` for numerical guard margin so zero remains feasible. Include TRUE and optionally prediction frontiers from all relations. Prediction protection has no edited-query exemption; conflicts with adding a new Top10 answer must be measured as utility loss.

Solve `min 0.5||delta-proposal||²` subject to these halfspaces. Cauchy pruning and deterministic cutting planes accelerate the solve; the original full constraint system must pass primal1e-10, nonnegative dual, stationarity/complementarity1e-6 and projection norm bound1e-8. Cap cutting rounds16, add at most64 rows/round, SLSQP500iterations/ftol1e-10. No tolerance/search expansion following results. Missing/failed certificate yields the zero update, never an unsafe analytic fallback.

Fixed entity radius is maxL2 over the old checkpoint plus random expansion before consulting new training facts. If needed, shorten the certified update along the segment from the current vector to its proposal endpoint to stay within that radius. Scaling towards zero preserves all nonnegative-slack halfspaces. Radius scaling is a separate replayed operation and does not inherit the unconstrained QP optimality certificate. Exact incremental guard counts check every committed shared vector; a proposed protected-set loss yields zero update. Full terminal guard recount and exact frozen oldE/R checks are mandatory.

## Evaluation, utility and limitations

Endpoint is fixed; no new model/config selection on TEST. Common old/new VALID uses seed124/1024 per split, with full final pool. Record source-proposal and postprocessed metrics on the same IDs and all-seed paired effects. Prespecified development utility screen: mean newVALID MRR at least98% of source proposal, and mean oldVALID Top10 loss at most75% of source proposal with all three seeds non-worse on old loss. If baseline old loss is zero, zero equal loss does not count as an improvement. This screen is descriptive with three seeds, not a significance/SOTA test. Final common TEST evaluation is reported for every frozen arm, including screen failures; never omit failure arms to claim superiority.

Log solver failures, zero updates, radius factors, primal/KKT maxima, frontier and pruned counts, retained TRAIN guard changes, update coverage, proposal and postprocessing costs, peak GPU memory, source hash and real query IDs. All initial-protected TRUE members must remain retained for QPTrue/QPPred arms; all prediction members additionally for QPPred. Neither guarantee applies to non-QP controls or unprotected queries. A no-update result with failed utility gate is NO_GO, even if it has perfect registered-set retention.

## Independent controls and audit

Synthetic control011 uses128 independently generated2D/3D cases plus duplicate/zero/dependent constraint cases. Exhaustive active-set enumeration is an independent optimum oracle, not a restatement of the SLSQP implementation. Require all ordinary/degenerate cases certified and vector error<=1e-7. Explicit one-iteration failure must produce exactly zero;256 radius cases include proposal magnitudes up to1e28 and halfspace checks after scaling. No graph facts, VALID or TEST are read by controls. Failure records remain; passing synthetic controls is not graph utility evidence.

Full graph auditor must reconstruct every entity step from original source vectors, guard masks, sparse dual witnesses and radius operation; verify all3753 IDs, frozen oldE/R, final guard recount, source/receipt/checkpoint hashes and complete common query identities. Use independent64 full-candidate scoring/aggregate auditor in addition. Audit limitations remain explicit; not independent optimizer retraining. Promote only audited lightweight evidence, never raw vectors/logs/checkpoints.
