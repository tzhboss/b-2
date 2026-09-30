# b-2

Long-term research repository with explicit experiment provenance and scientific evidence governance.

## Start here

- `AGENTS.md` — agent and engineering operating rules.
- `PROJECT_CONTEXT.md` — stable research context and source-of-truth model.
- `CURRENT_STATE.md` — short current project state.
- `docs/evidence_policy.md` — canonical execution / validity / decision semantics, results promotion, and Git-retention rules.
- `docs/experiment_workflow.md` — canonical experiment lifecycle.
- `docs/repository_structure.md` — canonical directory and file responsibilities.
- `EXPERIMENTS.md` — human-readable derived experiment index; never the machine source of truth.

## Experiment identity

Formal experiments use `EXP-YYYYMMDD-NN`. Each experiment's machine metadata lives only in:

`experiments/<EXPERIMENT_ID>/experiment.yaml`

Use `experiments/_TEMPLATE/` to register a real experiment. Do not create an experiment entry until there is a real research question and protocol.

## Evidence boundary

Raw runtime outputs belong under `artifacts/<EXPERIMENT_ID>/` (or other server-side artifact storage) and are not committed by default. Per-sample OOF/test predictions, embeddings, checkpoints, and logs stay outside Git. Only audited, lightweight evidence is promoted to `results/<EXPERIMENT_ID>/` with a results manifest.

For execution details, read the canonical policy documents rather than duplicating their rules here.
