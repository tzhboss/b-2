# Figure Captions

## Figure 1 — Target-reference matching

**Target-reference matching with semantically aligned Pitch+Rate.**
Mean speaker-balanced CCC difference between Relative and Absolute representations as the
between-speaker component of the target is increased from λ=0 (pure within-speaker target) to
λ=1 (raw target). Curves show Linear Ridge results for MSP-Podcast and IEMOCAP Arousal/Dominance.
Relative is favored at the within-speaker endpoint in all conditions, and its advantage decreases
as between-speaker target structure increases. Confirmatory slope tests are significantly negative
for both targets in both corpora under both Linear and Quadratic Ridge.

## Figure 2 — K-shot deployment

**Label-free K-shot speaker-reference estimation on MSP-Podcast.**
Speaker-balanced CCC of K-shot Hybrid Arousal and Dominance models as the number of unlabeled
enrollment utterances per speaker increases. Dashed lines show the diagnostic marginal-oracle
Hybrid performance. Downstream rows are fixed across K. At K=20, Arousal is within 0.0043 CCC of
the oracle and Dominance is effectively at oracle performance.

## Figure 3 — State/trait reversal

**Speaker-relative pitch redistributes state- and trait-related information.**
Relative-minus-Absolute Macro-F1 for categorical emotion and gender prediction on ESD, MEAD, and
RAVDESS. Relative pitch improves emotion classification across all three corpora while strongly
reducing gender prediction, motivating the view of normalization as an information transformation
rather than a universally superior representation.

## Figure 4 — WavLM reference-frame decodability

**Multiple prosodic reference frames remain decodable across WavLM layers.**
Cross-validated R² for Absolute pitch, speaker-relative pitch, and implied speaker baseline from
frozen WavLM-large hidden representations on ESD. Relative-pitch accessibility is strongest in
early/middle layers and decreases toward the final layers, while speaker-baseline information
remains strongly decodable.
