# Native GPU delta precision qualification of unchanged KGEditor artifacts

Preserve014/017, original failed43 CPU audit, weights, hypernetwork, data,
training selection and all evaluation arrays. Diagnostic on43 measures max CPU
delta error1.628324389e-5 against the original1e-5 tolerance; native GPU replay
of the same two saved steps has error0. No original CPU gate is relaxed or
rewritten. No retraining, new checkpoint, grid or positive-result selection.

031 applies one uniform precision qualification to all three existing seeds:
replay the original two native steps on GPU float32 with TF32 disabled, same
original1e-5 absolute delta tolerance, clean base/storage separation and full
TRAIN coverage/parameter scope/hash checks. Original immutable helper checks
inputs, before states and native deltas; a process-local GPU adapter transfers
its batches/deltas and preserves every helper assertion. No source module file
or other process is modified. Both original CPU passing controls are included.

Then execute the original CPU32 initial and64 terminal whole-candidate fixture
replays with unchanged1e-4 logit tolerance and exact ranks, all74428/12406
query identities, metrics, denominators and file hashes. Save qualification
receipts/audits in a separate directory and ensure original audits/checkpoints
remain byte-identical. A qualification failure remains failed, never replace
CPU rank assertions with GPU assertions or increase tolerances.

This is a qualified native-GPU-delta/CPU-inference audit. It does not turn the
original43 CPU-delta audit into PASS or certify all intermediate updates via
independent replay. Cumulative online use is a common-task adaptation, not the
author original protocol. Its empirical collapse cannot be extrapolated to
the author's original task. If promoted, show precision qualification and the
original failure prominently, retaining the same three method slots.
