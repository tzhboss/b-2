# KGEditor common novel-entity adaptation

## Fixed task and scientific scope

FB_CKGE snapshot0 to snapshot1; all 3753 new entities, 18607 original new TRAIN facts (37214 directed queries), 11258 final candidates. Seeds17/29/43. Old/new TEST and raw strict-greater ranking are exactly the common comparison. This is a disclosed author-method task adaptation and exposed development comparison, not an original-protocol replication or heldout superiority test.

The scientific foundation is the separately audited clean EXBert20-epoch old-graph model of LMBASE004. Language pretraining and public descriptions are additional architecture information. Full-graph KGE-pretrained author weights are prohibited. New entity vocabulary rows remain their original random initialization during base fitting.

## Native meta training

Retain `lm_editor_bridge_v2.train_meta` and the native BaseKEModule loss, OneShotLearner and optimizer. Fixed20epochs on6174 old-TRAIN proxy edit directions, with1024 disjoint old-TRAIN proxy validation directions; remaining216086 old-TRAIN directions provide stable samples. None of these proxy edit facts fitted the clean base. Two edits plus eight stable facts per step. First best half editMRR plus half stableMRR over the old7505 candidate pool selects the learner. Max_scale1, original RMSprop/alpha/scheduler and native loss retained. No new TRAIN or TEST facts enter meta fitting/selection. Registered18000s meta budget; failures remain failures.

Base and learner checkpoints are separate. CPU learner conditioner storage is explicitly cloned before loading to reproduce the disjoint storage of native GPU traversal. No restoration may overwrite base scientific embeddings. Native CPU/GPU fixture scores within1e-4 and exact strict ranks are required.

## Online task adaptation

Expand inference to all11258 entities. Uniform different wrong labels use fixed TRAIN-only corruption seed20261013 and native condition text/tokenizer. No prediction-derived or TEST-derived conditions. A torch seed+71 permutation visits each directed new TRAIN query exactly once in batches of two. Eight old baseTRAIN stable examples accompany every batch. The learner produces the original four extension-weight deltas with max_scale1. Apply these deltas cumulatively to a single actual shared EXBert, used for head and tail queries.

Native author evaluation is independent edits. Cumulative application is a substantive task adaptation; it may fail or learn poorly. Online learner optimization, clipping, hidden rescaling, rollback, and data-dependent step rejection are absent. The eight stable examples are input context, not an extra online protection loss. Require finite deltas/model at every step, identical learner parameters, and bitwise equality of every noneditable scientific state tensor to the clean base. One full sweep and18000s budget, not equal SGD epoch/cost. Record elapsed base, meta, online and evaluation separately.

The full endpoint is fixed, with no online checkpoint selection. Old/new VALID each uses the same common seed124 and1024 query identities for diagnostic reporting only. No extension or tuning on observed TEST results. Short native meta/online preflight is a gate, never a completed method.

## Common evaluation and audit

Only after full training has completed: old TEST74428 and new TEST12406 directions. Compute old-before on7505 and11258 pools, and actual final shared-model old/new-after on11258. Use rank=1+count(score>gold), no filtering. Report MRR, Hits1/10, old Top10 loss, same11258-pool loss and mean signed displacement. Stream logits rather than persist every full score matrix.

Audit complete TRAIN coverage/order, raw input/query identities, source/output/checkpoint hashes, immutable parameter scope, deterministic old TRAIN stable sampling, and two consecutive native proposal witnesses independently on CPU (delta tolerance1e-5). Re-encode all64 final fixture queries, replay64 final and32 initial full11258 score rows on CPU, require1e-4 score tolerance and exact ranks, and recompute all reported full-array aggregates. These audits are not independent full optimizer retraining. Failures retain all original artifacts and never receive numeric formal-table promotion.

Use live peer GPU claims for LMBASE004, rawRank005 and boundedRank007 and the shared six-worker CPU audit limit. Wait for actual audited base checkpoints; do not restart or alter running frozen implementations. Formal results include only audited lightweight evidence. Raw arrays/checkpoints/logs remain remote.
