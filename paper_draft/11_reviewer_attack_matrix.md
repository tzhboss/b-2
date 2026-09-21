# Reviewer Attack Matrix

## Attack 1: "Speaker normalization for arousal is old."

Valid criticism.

Closest evidence:
- Bone et al. 2012 already use neutral speaker baseline prosody for arousal.
- Busso et al. 2013 normalize speaker variability for emotion recognition.

Response:
Do not claim normalization novelty. Bone et al. 2012/2014 already provide speaker-baseline,
speaker-relative, cross-corpus arousal scoring, including limited-reference conditions.
Do not claim target-centering novelty either; within-person/between-person centering is standard in
multilevel affect research.
Claim the controlled **coupling**:
the same acoustic representation changes utility as target between-speaker structure is
systematically manipulated.

Evidence:
EXP-20260921-12 and EXP-20260921-06.

## Attack 2: "Your result is just because MSP has speaker leakage."

Response:
Primary splits are speaker-disjoint.
The center permutation uses separate train/test speaker derangements.

Evidence:
Experiment audits report zero speaker overlap.

Caveat:
IEMOCAP session-level heterogeneity is real and is disclosed.

## Attack 3: "Relative and Absolute are algebraically related; a nonlinear model should recover both."

Response:
The comparison tests accessible information under fixed low-dimensional representations, not
information-theoretic impossibility.
The primary slope survives both linear and quadratic Ridge.
Hybrid is included to expose baseline explicitly.

Evidence:
EXP-20260921-04 and EXP-20260921-12.

## Attack 4: "You chose a target transformation that guarantees your result."

Response:
The intervention changes only target between-speaker structure; the predicted direction is not a
mathematical identity because acoustic baseline need not correlate with target speaker offsets.
IEMOCAP shows much smaller slopes than MSP, and Valence is near zero, demonstrating that the effect
magnitude is empirical rather than mechanically guaranteed.

Additional mechanism evidence:
Correct-speaker center permutation strongly affects raw MSP A/D but not residual targets.

## Attack 5: "Full-speaker center uses test-speaker information."

Response:
Correct for the diagnostic oracle; this is explicitly scoped as mechanism analysis.
Deployment experiments estimate centers from separate unlabeled enrollment utterances excluded from
downstream rows.

Evidence:
EXP-20260920-19 and corrected fixed-pool EXP-20260920-21.

## Attack 6: "K-shot personalization is also old."

Valid criticism.

Response:
K-shot enrollment is not claimed as novel.
It is a deployment validation of the reference-frame mechanism.

Closest prior:
Triantafyllopoulos & Schuller 2024 and other personalization work.

## Attack 7: "Why use simple Ridge instead of a modern SER model?"

Response:
The primary contribution is a controlled representation/target analysis.
Simple fixed models reduce optimization and architecture confounds.
The primary mechanism is replicated under linear and quadratic models.
WavLM probing separately shows that multiple reference-frame variables remain decodable in a
modern SSL encoder.

Possible future extension:
Test target-reference intervention in an end-to-end model with explicit bottleneck controls.

## Attack 8: "IEMOCAP has only 10 speakers."

Valid limitation.

Response:
Treat IEMOCAP as external directional confirmation, not large-speaker inference.
MSP supplies large-speaker evidence.
Strict session-disjoint IEMOCAP sensitivity is explicitly heterogeneous and not used as primary
confirmation.

## Attack 9: "Your IEMOCAP loudness feature is unclear."

Already resolved.

Response:
Main confirmatory experiment excludes all loudness fields in both corpora.
Use Pitch + Speaking Rate only.

Evidence:
EXP-20260921-09 rejected loudness semantic robustness.
EXP-20260921-10 and EXP-20260921-12 confirm Pitch+Rate results.

## Attack 10: "Gender is not equivalent to speaker identity."

Agree.

Response:
Gender is only a strong stable-trait motivation.
A content-disjoint speaker-ID experiment shows only modest Relative suppression, and the manuscript
states this boundary explicitly.

Evidence:
EXP-20260921-11.

## Attack 11: "Why does Valence not follow the story?"

Response:
The paper does not claim universal VAD symmetry.
Low-dimensional acoustic prosody is weak for Valence in these data.
Prior work also reports strong speaker dependence and difficulty of acoustic Valence prediction.
Valence is treated as a boundary/negative-control dimension.

## Attack 12: "Session-disjoint IEMOCAP result is not significant."

Agree and disclose.

Response:
The original session analysis had pseudo-replicated deterministic seeds and was marked invalid.
The corrected five-session sensitivity shows negative mean slopes but substantial sign
heterogeneity. It is not used as confirmatory evidence.
