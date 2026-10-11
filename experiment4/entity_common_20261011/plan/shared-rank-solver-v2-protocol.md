# Immutable competitors in the local rank witness

025 original synthetic solver controls passed but fixed proposal witnesses
unnecessarily rejected the frozen-old control. Keep original code/config and
outcome unchanged.025R2 preserves all derivative, objective, shared-rank,
absolute movement and budget tests, and changes the witness construction.

If the query head and all reference gold rows are immutable for a step, any
immutable nonreference candidate already strictly above the current lowest
reference must remain in the allowed competitor set. Fill remaining slots in
the previous deterministic proposal order, deprioritizing adjustable old
competitors. For changed head/reference queries all scoring is variable and
the proposal witness remains conservative. If forced competitors outnumber
the worst-case allowance, all forced competitors remain allowed; the witness
alone then is not a sufficient rank certificate. Every candidate still needs
the entire actual reference scope acceptance check. Do not claim a rank
guarantee from these working constraints without the complete rank cache.

The frozen-old toy must now optimize its new vector up to the safe boundary
instead of falling back unchanged. It still cannot insert the new answer into
Top10 while the old rank10 answer and all nine immutable competitors remain.
Joint redistribution remains possible. Same64 finite-difference cases,
analytic joint optimum, independent scalar full-reference ranks, whole-cache
replay, forced zero-cut fallback and fixed absolute radii. Promotion requires
independent saved-model audit. This is implementation correctness, not graph
method utility, novelty, or completion of the broader research goal.
