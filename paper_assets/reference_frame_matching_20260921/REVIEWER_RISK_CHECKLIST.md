# Reviewer Risk Checklist

## High priority

### 1. "This is just old speaker normalization"

Response:
No. Prior work already establishes speaker normalization and speaker-relative arousal.
The contribution must be framed as the conditional rule governing whether speaker baseline should
be removed or preserved, demonstrated by direct target intervention.

Evidence:
EXP-20260921-12 and EXP-20260921-06.

### 2. "Your speaker-relative features leak test-speaker statistics"

Response:
Oracle-center experiments are explicitly labeled mechanism diagnostics.
Deployment is separately evaluated with label-free K-shot enrollment and disjoint downstream rows.

Evidence:
EXP-20260920-19 and corrected fixed-pool EXP-20260920-21.

### 3. "The result may depend on one regression model"

Response:
The primary Pitch+Rate target-reference slopes replicate under both Linear and Quadratic Ridge.

Evidence:
EXP-20260921-12.

### 4. "IEMOCAP loudness semantics are unclear"

Response:
Agree. The loudness robustness test failed and IEMOCAP loudness is excluded from the primary
confirmatory analysis. The main cross-corpus result uses Pitch+Rate only.

Evidence:
EXP-20260921-09 and EXP-20260921-12.

### 5. "IEMOCAP has only 10 speakers / 5 sessions"

Response:
This is a real limitation.
Speaker-disjoint results are consistent, but the stricter five-session sensitivity is heterogeneous
and is not claimed as confirmatory evidence.

Evidence:
EXP-20260921-13 corrected session-level sensitivity.

### 6. "Why is Valence weak?"

Response:
Low-dimensional prosody is known to be less predictive of Valence than Arousal/Dominance.
Our own experiments confirm that Valence is near a negative-control dimension for this mechanism.
Do not force a universal VAD claim.

### 7. "Gender is not speaker identity"

Response:
Correct.
Gender is presented only as a strong stable-trait example. A separate speaker-ID audit shows
Relative prosody suppresses identity only modestly.

Evidence:
EXP-20260921-11.

### 8. "WavLM already encodes everything, so explicit reference frames are irrelevant"

Response:
WavLM decodes Absolute, Relative, and baseline information simultaneously, but explicit
Relative-vs-Absolute augmentation after WavLM does not show a universal emotion advantage.
This supports the information-reorganization view rather than a simplistic invariance claim.

Evidence:
EXP-20260920-03 and EXP-20260920-04.

## Medium priority

### 9. "Target intervention is artificial"

Response:
It is intentionally controlled.
Its purpose is to isolate mechanism by changing only between-speaker target structure while
holding inputs/model fixed. Raw targets and fully centered targets are both real endpoints of this
continuum.

### 10. "Full-speaker target mean uses future labels"

Response:
Yes, in the mechanism experiment.
It is a diagnostic decomposition, not a deployment procedure. Deployment claims are confined to
acoustic center estimation, not target centering.

### 11. "Why use simple Ridge models?"

Response:
The core question concerns information accessibility and causal structure, not leaderboard
performance. Simple fixed models minimize hidden optimization confounds. Quadratic Ridge confirms
that the mechanism is not restricted to purely linear access.

### 12. "Could results be annotation-noise artifacts?"

Response:
Low-disagreement MSP subsets retain the same Arousal/Dominance reference-frame effects.

Evidence:
EXP-20260920-18.

## Claims that should never appear without qualification

- Relative prosody is universally better for emotion recognition.
- Speaker variability should always be removed.
- Speaker-relative normalization removes speaker identity.
- IEMOCAP loudness confirms the mechanism.
- K=20 is universally optimal.
- WavLM is speaker-invariant.
- The current work invents speaker normalization.

### 13. "Fold/seed bootstrap treats non-independent units as independent"

Response:
The main Pitch+Rate uncertainty analysis was re-run using complete OOF predictions and
speaker-cluster bootstrap. MSP slopes remain strongly significant. IEMOCAP slope CIs widen and
cross zero with only 10 speakers, and the manuscript should state this explicitly.

Evidence:
EXP-20260921-15.

### 14. "The mechanism is only an artifact of subtracting average pitch"

Response:
No. On MSP the same target-reference coupling appears for F0 variability, pause ratio, and voiced
ratio, with all Arousal/Dominance slope CIs below zero.

Evidence:
EXP-20260921-14.

### 15. "The effect is a trivial artifact of subtracting handcrafted prosodic means"

Response:
The same target-reference preference reversal appears when reference frames are constructed
directly in 1024-dimensional frozen WavLM embedding space at both layer 12 and layer 24.

Evidence:
EXP-20260921-16.

### 16. "High-dimensional WavLM Hybrid Ridge is ill-conditioned"

Response:
The primary registered WavLM claim is Relative-minus-Absolute, not a small Hybrid ranking.
All predictions are finite and the registered Relative-minus-Absolute speaker-cluster effects are
large and significant. A solver-stability check should be reported before submission if it changes
any main conclusion.

### 15. "The effect is a trivial artifact of subtracting handcrafted prosodic means"

Response:
The same target-reference preference reversal appears when reference frames are constructed
directly in 1024-dimensional frozen WavLM embedding space at both layer 12 and layer 24.

Evidence:
EXP-20260921-16.

### 16. "High-dimensional WavLM Hybrid Ridge is ill-conditioned"

Response:
The primary registered WavLM claim is Relative-minus-Absolute, not a small Hybrid ranking.
All predictions are finite and the registered Relative-minus-Absolute speaker-cluster effects are
large and significant. A solver-stability check should be reported before submission if it changes
any main conclusion.

### 15. "The effect is a trivial artifact of subtracting handcrafted prosodic means"

Response:
The same target-reference preference reversal appears when reference frames are constructed
directly in 1024-dimensional frozen WavLM embedding space at both layer 12 and layer 24.

Evidence:
EXP-20260921-16.

### 16. "High-dimensional WavLM Hybrid Ridge is ill-conditioned"

Response:
The primary registered WavLM claim is Relative-minus-Absolute, not a small Hybrid ranking.
All predictions are finite and the registered Relative-minus-Absolute speaker-cluster effects are
large and significant. A solver-stability check should be reported before submission if it changes
any main conclusion.

### 13. "Fold/seed bootstrap treats correlated units as independent"

Response:
The main inferential analysis has been replaced by a speaker-cluster bootstrap on complete OOF
predictions. MSP remains strongly significant under speaker-level resampling. IEMOCAP is correctly
reported as directionally consistent but underpowered for slope inference with only ten speakers.

Evidence:
EXP-20260921-15.

### 14. "The effect is a handcrafted-feature arithmetic artifact"

Response:
The same target-reference intervention was repeated directly in frozen WavLM embedding space.
At both layer 12 and layer 24, Arousal and Dominance show significantly negative
Relative-minus-Absolute slopes under speaker-cluster bootstrap, with Relative favored at lambda=0
and Absolute favored at lambda=1.

Evidence:
EXP-20260921-16.

### 15. "The effect only applies to mean pitch/loudness/rate levels"

Response:
MSP F0 variability, pause ratio, and voiced ratio all show significant target-reference coupling.
The mechanism therefore extends to expressive range and temporal/voicing structure.

Evidence:
EXP-20260921-14.
