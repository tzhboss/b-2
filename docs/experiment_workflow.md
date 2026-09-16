# Experiment Workflow

This file is the canonical home for the experiment lifecycle and execution order.

## 1. Define the research question

State a scoped question that can be answered by evidence. Do not create a formal experiment merely to reserve an ID.

## 2. Register the protocol

Create or select a protocol under `configs/protocols/`. The protocol defines what evidence would be sufficient: datasets and forbidden uses, controls, changed factors, baselines, metrics, splits, seeds, checkpoint selection, acceptance/rejection/inconclusive criteria, required outputs, and required audits.

## 3. Register the formal experiment

Assign the next real ID in `EXP-YYYYMMDD-NN` format and create:

`experiments/<EXPERIMENT_ID>/experiment.yaml`

Copy the report and audit templates into the same directory. Fill the registered hypothesis before looking at results. Link parent or superseded experiments only when the relationship is real.

## 4. Create the one-run config

Create `configs/experiments/<EXPERIMENT_ID>.yaml`. This records how this specific run should execute. Do not move protocol acceptance logic into the run config.

## 5. Execute

Codex or another execution agent performs actual code/runtime work. Runtime facts such as commit, branch, seed, environment, checkpoint, outputs, metrics, and hashes must come from observation, not inference.

## 6. Audit

Complete `experiments/<EXPERIMENT_ID>/audit.md` against the protocol and actual outputs. Update validity only from audit evidence. A failed or invalid experiment remains in repository history.

## 7. Promote results

Promote only audited, lightweight evidence to `results/<EXPERIMENT_ID>/`. Create `_results_manifest.yaml` and record provenance for every promoted file.

## 8. Interpret

Complete `report.md` after audit. Keep registered hypothesis separate from results, observation, supported claim, unsupported stronger claim, and post-experiment interpretation.

## 9. Update derived state

Refresh `EXPERIMENTS.md` and the short `CURRENT_STATE.md` only after the authoritative experiment YAML is updated. Do not create a second manually maintained status source.
