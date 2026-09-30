# Evidence Policy

This file is the canonical home for experiment status semantics, evidence validity, supersession, and formal results promotion.

## Orthogonal status axes

Each formal experiment records three independent status axes in `experiment.yaml`.

### execution

Describes runtime progress only:

- `planned`
- `running`
- `completed`
- `failed`

### validity

Describes whether the evidence is trustworthy under the governing protocol:

- `unchecked`
- `valid`
- `invalid`

Completion alone never establishes validity.

### decision

Describes the scientific decision under the protocol:

- `pending`
- `pass`
- `fail`
- `inconclusive`

Do not encode checkpoint names, selection state, execution state, or supersession into `decision`.

## Supersession is not invalidity

Historical replacement is represented only with:

- `supersedes`
- `superseded_by`

A superseded experiment can remain valid evidence. Supersession records lineage, not evidence quality.

## Registration versus interpretation

`registered_hypothesis` is fixed before results are inspected. It must not be rewritten afterward to match observed outcomes.

Scientific reports must distinguish:

1. Registered Hypothesis
2. Results
3. Observation
4. Supported Claim
5. Unsupported Stronger Claim
6. Post-experiment Interpretation
7. Decision
8. Next Step

`Observation` states what the data directly show. `Supported Claim` is the strongest statement justified by audited evidence. `Unsupported Stronger Claim` records tempting but currently unjustified conclusions. `Post-experiment Interpretation` may discuss mechanisms or explanations but must remain identifiable as interpretation.

## Artifact and result boundary

`artifacts/<EXPERIMENT_ID>/` contains runtime material such as checkpoints, logs, raw predictions, embeddings, and temporary outputs. It is not committed by default.

`results/<EXPERIMENT_ID>/` contains lightweight, formal research evidence only after audit. Each promoted result set requires `_results_manifest.yaml` linking the experiment, runtime commit, protocol, config, audit, and promoted files.

## Promotion rule

An output may enter formal `results/` only when:

- the experiment identity and runtime provenance are known;
- the required audit is complete;
- protocol violations and missing outputs are recorded;
- the evidence selected for promotion follows the protocol's selection rule;
- the results manifest records provenance.

If these conditions are not satisfied, keep the output in artifacts or leave the result unpromoted.

## Git retention and result granularity

Git stores the scientific record, not the complete runtime dump. The default retention rule is semantic rather than extension-based.

Commit to Git when an output is lightweight and necessary to understand, audit, or cite the experiment, including:

- aggregate Overall / Between / Within metrics and confidence intervals;
- fold-, seed-, speaker-, or condition-level summaries needed for audit;
- selection, training, leakage, mapping, alignment, and runtime-integrity audits;
- experiment reports, manifests, protocol/config files, and paper-facing tables/figures.

Do not commit outputs whose primary role is to reproduce later computation rather than document the conclusion, including:

- per-utterance OOF/test predictions and prediction shards;
- embeddings, hidden states, feature caches, checkpoints, optimizer states, and model caches;
- runtime logs and temporary files;
- raw/copyrighted datasets or audio.

These files remain server-side under `artifacts/`, `logs/`, or another experiment-scoped external storage location. If a large artifact must be published, use an external artifact store, release asset, or archival service and record its identifier/hash in the experiment metadata or report rather than placing the payload in Git history.

Size is a secondary guardrail: review any newly tracked result file above 10 MB. A file near or above GitHub's 100 MB object limit is prohibited from normal Git history even if it is reproducible.

Compiled manuscript PDFs may be retained for explicit submission/release milestones, but transient LaTeX/render build products are not research evidence and remain ignored.
