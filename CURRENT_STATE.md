# CURRENT_STATE.md

## Phase

Reference-target diagnostic registered; execution pending.

## Active research question

Does label-free K-shot relative pitch work because it estimates a speaker's unlabeled marginal
pitch center rather than the neutral-derived reference?

## Valid evidence

- EXP-05 shows unseen-speaker K-shot relative pitch strongly outperforms absolute pitch and
  reaches oracle-relative downstream performance by K=10, despite non-monotonic error relative
  to the neutral-derived oracle baseline.

## Next legal step

Run EXP-20260920-06 using the exact EXP-05 enrollment/split procedure and compare K-shot estimates
against neutral and full marginal speaker references.
