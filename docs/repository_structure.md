# Repository Structure

This file is the canonical home for directory and file responsibilities.

```text
.
├── README.md
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── CURRENT_STATE.md
├── EXPERIMENTS.md
├── docs/
│   ├── evidence_policy.md
│   ├── experiment_workflow.md
│   └── repository_structure.md
├── configs/
│   ├── protocols/
│   │   └── _TEMPLATE.yaml
│   └── experiments/
│       └── _TEMPLATE.yaml
├── experiments/
│   └── _TEMPLATE/
│       ├── experiment.yaml
│       ├── report.md
│       └── audit.md
└── results/
    └── _TEMPLATE/
        └── _results_manifest.yaml
```

`artifacts/` is a runtime location and is ignored by Git by default, so it does not need a committed placeholder.

## Responsibilities

- `configs/protocols/`: reusable evidence requirements for answering research questions.
- `configs/experiments/`: concrete parameters for one run.
- `experiments/<ID>/experiment.yaml`: sole machine source of truth for experiment metadata/status.
- `experiments/<ID>/audit.md`: execution/protocol compliance record based on observed facts.
- `experiments/<ID>/report.md`: scientific interpretation of audited results.
- `results/<ID>/`: audited, lightweight evidence plus provenance manifest.
- `artifacts/<ID>/`: raw runtime outputs, usually not committed.

Create `src/`, `scripts/`, and `tests/` only when real implementation work requires them. Create `configs/prompts/` or `configs/data/` only after the project has a concrete need for those concepts. Their absence at bootstrap is intentional.
