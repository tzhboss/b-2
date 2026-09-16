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
