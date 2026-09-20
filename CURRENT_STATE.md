# CURRENT_STATE.md

## Phase

Pitch, loudness, and speaking-rate reference-frame experiments completed across controlled and
natural corpora.

## Main valid findings

- Pitch: strong task-, corpus-, and emotion-dependent Absolute/Relative effects; MSP aggregate
  reversal is dominated by neutral and stable speaker-baseline information.
- Loudness: class-level heterogeneity generalizes, but the preferred frame can differ from pitch
  for the same corpus/emotion. MSP neutral prefers Relative loudness while preferring Absolute pitch.
- Speaking rate: class-level heterogeneity also generalizes. MSP shows a large split: angry favors
  Relative rate, while neutral and sad favor Absolute rate and recover when speaker rate baseline
  is restored.
- Across loudness/rate, nine of ten corpus-by-attribute cells have >0.03 classwise
  Relative-minus-Absolute F1 range.
- Baseline addition causes resolved class-specific gains/losses in multiple corpora and attributes.
- Fixed absolute conflict thresholds do not yield broad enough coverage for a confirmatory
  loudness/rate conflict-set claim.

## Paper-level interpretation

The evidence now supports Task × Attribute × Corpus/Domain × Emotion Class interactions in
prosodic reference-frame utility. Speaker normalization is not a universally beneficial operation;
it redistributes information between within-speaker deviation and stable speaker baseline, and
the usefulness of those components changes by attribute and target class.

## Next legal step

Build a compact unified statistical model / effect-size table across pitch, loudness, and rate,
then run a preregistered leave-one-corpus-out analysis to test whether reference-frame preference
can be predicted from attribute/class statistics rather than selected post hoc.
