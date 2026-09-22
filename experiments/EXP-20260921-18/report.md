# Experiment Report — EXP-20260921-18

Unlabeled K-shot enrollment recovers the WavLM target-reference effect once the enrollment set is moderately sized.

At K=20, all four layer-by-target Relative-minus-Absolute slope means are negative. Three of four pure within-speaker (lambda=0) endpoint effects are positive; the exception is layer-12 Dominance, which is essentially zero.

At K=50, all four slope signs match the full-speaker oracle. Three of four absolute slope gaps are <=0.03; layer-24 Dominance is the only cell above that threshold (0.0433).

Recovery is not monotonic at very small K: some K=1/K=5 cells remain unstable or reverse direction. The deployment claim should therefore be limited to moderate enrollment sizes (roughly 20-50 unlabeled utterances here), not arbitrary few-shot enrollment.

Decision: pass.
