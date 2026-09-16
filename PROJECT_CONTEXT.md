# PROJECT_CONTEXT.md

This file is the canonical home for stable research context and the repository source-of-truth model. Keep short-lived progress out of this file.

## Research objective

The specific scientific objective has not yet been registered in the repository. Add it here only when it is stable enough to guide multiple experiments.

## Core governance principle

> minimal structure, strong provenance, explicit scientific boundaries.

The repository must make it possible to distinguish proposed research intent, actual execution facts, audited evidence, and scientific interpretation without duplicating machine state.

## Source-of-truth model

- Experiment machine metadata and status: `experiments/<EXPERIMENT_ID>/experiment.yaml`.
- Formal evidence semantics and promotion rules: `docs/evidence_policy.md`.
- Experiment lifecycle: `docs/experiment_workflow.md`.
- Directory/file responsibilities: `docs/repository_structure.md`.
- Current active state: `CURRENT_STATE.md`.
- Human-readable experiment index: `EXPERIMENTS.md`, derived from experiment YAML and never authoritative.
- Git history records repository change history.

## Stable role split

- Research design/governance: research question, protocol, experiment relationships, evidence interpretation, next-experiment design.
- Execution/Codex: code and file changes, tests, runtime inspection, training/evaluation when explicitly tasked, artifacts, hashes, audits, and synchronization of observed facts.

No agent may convert a guess about an inaccessible runtime into a repository fact.

## Explicit non-goals at bootstrap

Until a concrete need appears, this repository does not define a claim registry, paper metadata system, dashboard, database, workflow engine, complex data schema, complex prompt schema, or experiment-specific code framework.
