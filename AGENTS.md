# AGENTS.md

This file is the canonical home for agent and engineering operating rules.

## Required operating rules

1. Never hard-code machine-specific absolute paths.
2. Express one-off experiment differences through config before creating new code.
3. Do not create a dedicated script merely because a new experiment exists.
4. Put reusable logic in `src/` only when reusable logic is actually needed.
5. Put CLI/orchestration in `scripts/` only when execution tooling is actually needed.
6. Put checks in `tests/` only when code or validation behavior exists to test.
7. `completed` does not imply `valid`.
8. Unaudited outputs must not be promoted into formal `results/`.
9. Failed experiments are retained; do not delete their records to make history look clean.
10. `experiments/<ID>/experiment.yaml` is the sole machine source of truth for experiment metadata and status.
11. Do not manually maintain the same experiment status in multiple files.
12. Codex/execution agents must not invent scientific conclusions.
13. ChatGPT/research agents must not invent runtime facts, server state, checkpoints, metrics, hashes, or artifacts.
14. Preserve registered hypotheses after results are visible; post-experiment interpretation belongs in `report.md`.
15. Keep protocol rules separate from one-run experiment configuration.
16. Keep raw artifacts separate from promoted research results.
17. Keep changes Git-traceable; do not overwrite unrelated user work.
18. Do not commit per-sample OOF/test prediction tables; keep them under server-side artifacts even when they are useful for later re-analysis.
19. Do not commit runtime logs, embeddings, model checkpoints, raw datasets, or raw audio.
20. Promote only lightweight audited evidence to `results/`: aggregate metrics, confidence intervals, component deltas, selection/training audits, manifests, and other reviewer-facing evidence.
21. Treat 10 MB as a review threshold for any new tracked result file; files approaching 100 MB require an external artifact store and must not enter Git history.
22. Before pushing, inspect staged files and large Git blobs; a successful experiment does not justify committing all of its runtime outputs.

## Minimal context for an ordinary experiment task

Read only what is needed:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `experiments/<ID>/experiment.yaml`
- the referenced protocol in `configs/protocols/`
- the referenced experiment config in `configs/experiments/`

Read `PROJECT_CONTEXT.md` or policy docs only when the task needs stable project context or governance interpretation.

## Execution boundary

Research agents define what should happen and what evidence would mean. Execution agents verify what actually happened and write only observed runtime facts. When runtime access is unavailable, leave runtime fields unknown rather than guessing.
