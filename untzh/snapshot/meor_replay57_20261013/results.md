# 057 MEOR adaptation versus replay

Four fresh paired arms, seed17, snapshot0→1, 20 epochs/740 steps. The MEOR aggregate and reference mean follow Equations12–15; stepcap is our explicit adaptation, not the author coefficient rule. Filtering uses cumulative TRAIN truths only; exposed VALID development evaluation, no TEST.

| Arm | New filtered MRR | Old head → new gold ratio to new_only | New head → old gold ratio | Both new ratio | Old VALID drops | Old TRAIN drops | Old VALID MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| new_only | 0.150805134 | 1.000000 | 1.000000 | 1.000000 | 7 | 5418 | 0.220174848 |
| replay_010 | 0.149749065 | 0.967516 | 0.996924 | 1.002597 | 4 | 4654 | 0.221220229 |
| replay_030 | 0.150994540 | 0.949041 | 1.012162 | 0.989879 | 3 | 3767 | 0.221781366 |
| replay_100 | 0.153509073 | 0.890150 | 1.042804 | 1.009809 | 2 | 2380 | 0.222396408 |
| replay | 0.153509073 | 0.890150 | 1.042804 | 1.009809 | 2 | 2380 | 0.222396408 |
| meor_stepcap | 0.150484475 | 0.979988 | 1.000268 | 1.008544 | 6 | 3235 | 0.220269799 |
| absolute_stepcap | 0.149064241 | 0.908578 | 1.001906 | 1.006172 | 3 | 1833 | 0.222100367 |
| replay_plus_meor_stepcap | 0.153299743 | 0.881903 | 1.042565 | 1.010000 | 2 | 1545 | 0.222542527 |
| meor_fixed_056 | 0.125024341 | 0.727967 | 0.842662 | 0.888420 | 32 | 2186 | 0.213452946 |
| absolute_fixed_056 | 0.148288853 | 0.907002 | 0.995648 | 1.005842 | 3 | 1787 | 0.222079136 |

Fresh replay reproduces the historical weight1 sampled softplus replay objective. replay_plus_meor_stepcap adds MEOR to that identical host, with the same replay tape. meor_stepcap and absolute_stepcap share the new-only host, complete candidate cache, and cap rule. The paper calls its zero-MEOR host Replay; project historical replay includes a separate sampled negative penalty. These names must not be treated as interchangeable.

Fixed-MEOR (056 structural), fixed-full-pool absolute and replay .1/.3/1 are reused historical context, not independent repeats.

## Fixed quality criteria

- meor_stepcap_vs_new_only: FAIL; failed=['new_mrr_all_and_directions_at_least_98pct', 'old_valid_drops_reduce_at_least_25pct']
- absolute_stepcap_vs_new_only: FAIL; failed=['new_mrr_all_and_directions_at_least_98pct', 'direction_hits10_at_least_95pct']
- meor_stepcap_vs_absolute_stepcap: FAIL; failed=['old_valid_drops_reduce_at_least_25pct', 'old_train_drops_no_worse']
- replay_plus_meor_stepcap_vs_replay: FAIL; failed=['old_valid_drops_reduce_at_least_25pct']

95% paired query bootstrap intervals are descriptive single-seed summaries; shared entities and multiple comparisons limit inference. Full metrics, individual ranks and gradient trajectories are retained. No tuning follows this comparison.

## Costs

Actual new run total: 116.095s; reused preparation historical cost: 585.799s. Shared reference preparation includes random references and is not a standalone method cost. No independent audit/archive/hard timeout.
