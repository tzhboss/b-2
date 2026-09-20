# Experiment Audit — EXP-20260920-12

## Completeness

- 75 unique corpus × attribute × emotion cells.
- Router choices are derived only from EXP-11 leave-one-corpus-out predictions.
- No adaptive router uses a model trained on its target corpus.
- Cell F1 outcomes reconstruct from EXP-09/10 summaries.

## All-cell results

| Policy | Mean F1 | Oracle regret | Gain vs better fixed |
| --- | ---: | ---: | ---: |
| Always Absolute | 0.1962 | 0.0344 | -0.0187 |
| Always Relative | 0.2149 | 0.0157 | 0.0000 |
| Labels-only router | 0.2080 | 0.0226 | -0.0070 |
| Stats-only router | 0.2246 | 0.0060 | +0.0096 |
| Combined router | 0.2186 | 0.0120 | +0.0037 |
| Oracle router | 0.2306 | 0.0000 | +0.0157 |

The stats-only router recovers about 61.5% of the oracle gain over the better fixed policy and
reduces oracle regret to about 38.5% of the better fixed policy's regret.

On the 43 resolved-effect cells, stats-only gains +0.0149 F1 over the better fixed policy.

## Registered criteria

- Practical routing: supported (+0.0096 > +0.005).
- Regret reduction: supported (0.00604 <= 0.75 × 0.01568).
- Corpus robustness: not supported; stats-only beats the better fixed policy in MELD and MSP,
  ties ESD, and is slightly worse in MEAD and RAVDESS.

## Verdict

- Validity: valid.
- Decision: mixed.
- Adaptive held-out-corpus routing improves aggregate performance and substantially reduces regret,
  but corpus-by-corpus robustness is incomplete.
