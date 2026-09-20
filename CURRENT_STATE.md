# CURRENT_STATE.md

## Phase

Held-out-corpus reference-frame routing completed and audited.

## Main finding

Stats-only leave-one-corpus-out routing improves mean F1 by +0.0096 over the best fixed policy and
reduces oracle regret by about 61.5%, but gains are not positive in every corpus.

## Next legal step

Run a nested confidence-aware router: choose confidence threshold and fallback using only inner
training-corpus cross-validation, then apply the selected policy to the untouched outer corpus.
