# Repository Artifact Policy

This repository stores the durable scientific record: code, protocols, experiment registration, audits, compact result summaries, and manuscript sources. It is not a bulk storage location for reproducible runtime artifacts.

## Commit by default

- Source code: `src/`, `scripts/`
- Experiment and protocol configs: `configs/`
- Experiment registry and experiment metadata: `EXPERIMENTS.md`, `experiments/**/experiment.yaml`
- Human-readable reports and audits: `report.md`, `audit.md`, reviewer-facing notes
- Compact machine-readable evidence:
  - `audit_summary.json`
  - `run_metadata.json`
  - runtime integrity metadata
  - metric, delta, CI, comparison, stability, selection, training-audit, and summary tables
- Project state and governance documents
- Manuscript source, bibliography, final paper tables, and canonical final figures

## Do not commit by default

- Raw/private/licensed datasets or audio
- Extracted SSL embeddings or feature caches
- Model checkpoints and framework caches
- Runtime logs
- Per-utterance / per-frame / per-timestep predictions
- Reproducible OOF/test prediction shards
- Temporary or intermediate files
- LaTeX build products and render-check images

Typical excluded result names include:

- `oof_predictions.parquet`
- `oof_*.parquet`
- `paired_oof*.parquet`
- `test_predictions.parquet`
- `audio_manifest.parquet`

These files should remain server-side under the experiment workspace or be published separately as a release/archive only when needed for external reproduction.

## Decision rule

Before committing a generated file, ask:

1. Is it required to understand, audit, or reproduce the scientific conclusion?
2. Is it a compact statistical summary rather than a row-level runtime artifact?
3. Would losing it destroy an experiment decision that cannot be reconstructed from code/config/seed/data?

If (1) and (2) are yes, commit it. If the file is large row-level output that can be regenerated, keep it server-side.

## Size rule

No ordinary Git blob should approach GitHub's 100 MB hard limit. Large artifacts belong outside normal Git history even when technically uploadable.

## Manuscript build rule

Track manuscript source and canonical figures. Ignore generated LaTeX products such as `.aux`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, temporary render checks, and routine compiled PDFs. A release PDF may be attached to a tagged release instead.

## Experiment minimum audit package

Each finalized experiment should retain at least:

- registered experiment metadata
- exact config/protocol
- run metadata
- audit/validity decision
- compact primary metrics
- paired deltas / uncertainty summaries when relevant
- report or interpretation for experiments used in the paper

This keeps the Git repository small while preserving the complete claim-to-evidence chain.
