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

## Version-control retention

`results/` is a curated evidence layer, not a mirror of every file produced by an experiment. Keep aggregate metrics, statistical summaries, audit tables, manifests, and paper-facing evidence there. Per-sample predictions (`oof_predictions.parquet`, `test_predictions.parquet`, model-specific `oof_*.parquet`, and paired OOF shards) remain server-side and are ignored by Git.

`artifacts/` and `logs/` are local/server runtime layers. Large embeddings, checkpoints, caches, raw predictions, raw data, and logs belong there or in another external artifact store. Record provenance, hashes, checkpoint identifiers, and regeneration commands in committed metadata instead of committing the large payloads.

Paper source and final figures are versioned. Transient LaTeX/render outputs are ignored; compiled PDFs are versioned only for explicit milestone/submission snapshots.
