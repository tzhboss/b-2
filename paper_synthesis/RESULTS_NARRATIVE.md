# Results Narrative Skeleton

## Result 1 — Relative normalization is not uniformly beneficial

In controlled affective corpora, speaker-relative pitch improves categorical emotion prediction
while substantially reducing gender information. This establishes a task-dependent reversal rather
than a universal normalization advantage. Per-emotion and per-attribute analyses further show that
reference-frame preference varies within the affect task itself.

## Result 2 — The target reference frame explains raw VAD differences

MSP and IEMOCAP differ sharply in between-speaker VAD structure. MSP Arousal and Dominance contain
large between-speaker components, whereas IEMOCAP contains much less. Consistent with this
difference, raw MSP VAD benefits from stable speaker baseline information, while IEMOCAP already
favors Relative features more strongly.

## Result 3 — Controlled intervention supports target-reference matching

When the between-speaker target component is increased continuously while acoustic inputs, folds,
and model are held fixed, Relative-minus-Absolute CCC decreases significantly for Arousal and
Dominance in both MSP and IEMOCAP. This pattern persists under both linear and quadratic Ridge,
showing that the result is not specific to one linear readout.

## Result 4 — Correct center identity is necessary for raw baseline utility

Permuting speaker centers across speakers preserves the center distribution but destroys the
speaker-center correspondence. This intervention removes a large fraction of raw MSP Arousal and
Dominance performance, while having negligible effect after the target is centered within speaker.
The utility therefore comes from the alignment between the stable acoustic baseline and the
speaker-level target component.

## Result 5 — Speaker reference can be estimated without labels

A small set of unlabeled enrollment utterances substantially reduces acoustic-center estimation
error. K-shot Hybrid improves raw VAD prediction and approaches the full-speaker oracle. Downstream
performance shows stronger diminishing returns than acoustic-center error, with K around 20
providing a practical operating point on MSP.

## Result 6 — External evidence survives conservative feature semantics

IEMOCAP source relative_db does not behave like a conventional absolute loudness measure, and
loudness-based claims are therefore excluded from the main evidence. Using pitch and speaking rate
only, the target-reference slopes remain significantly negative for Arousal and Dominance under
both linear and quadratic models, preserving the external replication without the ambiguous
feature.
