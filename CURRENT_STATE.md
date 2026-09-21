# CURRENT_STATE.md

## Phase

IEMOCAP loudness semantic robustness completed and rejected.

## Critical correction

The source field relative_db has undocumented semantics and is nearly unrelated to 20*log10(rms).
Replacing it with RMS dB reverses IEMOCAP Arousal/Dominance target-reference slopes. Therefore
IEMOCAP loudness and all-feature results that depend on relative_db are not suitable as main
evidence.

## Next legal step

Re-test IEMOCAP target-reference coupling using only semantically clear pitch and speaking-rate
features. If pitch+rate preserves negative Arousal/Dominance slopes, retain IEMOCAP as an
independent external validation without loudness.
