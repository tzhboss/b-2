# Experiment Report — EXP-20260921-17

The WavLM target-reference result is numerically solver-stable.

Replacing the default Ridge solver with LSQR at tolerance 1e-8 changes the four primary
Arousal/Dominance slopes by at most 1.65e-6 and the pure within-speaker endpoint effects by at most
1.01e-6.

All four slope signs, all four endpoint signs, and all speaker-cluster confidence conclusions are
unchanged.

Decision: pass.
