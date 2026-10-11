# Structural follow-up notes pending complete032 utility evidence

This is an exploratory interpretation draft, not a registered new method or
utility decision. Preserve027/030/032 source gates and failures. Complete the
three-seed fixed utility screen before selecting a follow-up implementation.

Remote raw evidence:
`artifacts/shared-parameter-incidence/20261011T072428Z/summary.json`.
The diagnostic reads only old/new TRAIN and currently audited joint artifacts.
It does not compute a score/rank counterfactual or read VALID/TEST. Input hashes
and pending source IDs are recorded. Only seed17 was audited at diagnostic time.

For seed17,64 old rows were selected and58 actually changed over116 steps.
The changed rows occur in6077 old TRAIN head facts and6077 tail facts, and in655
new TRAIN old-head/new-gold facts and655 reciprocal new-head/old-gold facts.
There are600 distinct new heads targeting these changed old rows. Their805
guarded head/relation keys cover805/27492=2.9281% of all58*474 model head/relation
combinations. This denominator includes unobserved combinations: it is not a
count of real old truths or realized TEST damage. Candidate/gold-role guards
remain active on every compiled query, so this is specifically head-key scope.

Audited exposed seed17 TEST results also show joint oldTop10 loss4.9922%,
versus source2.2447% and scoped-frozen1.4224%. New-head/old-gold MRR falls from
source0.247989 to joint0.220449. These separate observations do not prove that
the head-key scope or changed TRAIN answer rows caused those differences.
Complete032 VALID decisions remain authoritative for the frozen utility gate.

Mechanism questions for follow-up:

- Which losses follow changed old rows as heads, as gold answers, or as
  competing candidates, with the new vectors held fixed in an explicitly
  labelled endpoint counterfactual? That intervention differs from replaying
  the entire optimization trajectory.
- Does prediction preservation for all model relations of TRAIN-selected old
  heads improve this trade-off? Such extra predictions may be unsupported
  or conflict with new facts. Any newTRAIN query exemption must be explicit.
- Can a neutral new-row initial state retain more original prediction members
  than024's random expansion? Original prediction eligibility is only about
  15500/68334 complete query sets. Feasibility, negative-score boundaries and
  new-answer utility require measurement; no presumed global guarantee.

Do not increase the old budget/radius or search coefficients from these
observations. A proposed change needs a fixed mechanism, its control,
unchanged common evaluation, complete coverage disclosure and independent
audit.034 qualifies cached implementation equivalence only; it cannot turn
a negative032 method result into scientific success or establish novelty.
