# Experiment Report — EXP-20260923-10

## Result

The Dang et al. (Interspeech 2016) factor-analysis speaker-normalization mechanism was reimplemented on the unchanged common six-feature MSP absolute Arousal/Dominance task with a fixed Ridge readout. Evaluation uses 5-fold speaker-disjoint splits over three seeds and paired speaker-cluster bootstrap over 1,915 speakers.

Speaker-factor subtraction has a sharply asymmetric component profile at every preregistered latent dimension q in {1,2,3,5}: **Within-speaker CCC improves slightly, while Between-speaker CCC collapses, causing Overall CCC to decrease.**

Representative q=1 effects versus the unnormalized baseline:
- Arousal: Overall **-0.13804**, 95% CI [-0.15161,-0.12459]; Between **-0.32319**, [-0.34836,-0.29722]; Within **+0.01383**, [+0.01328,+0.01436].
- Dominance: Overall **-0.09296**, [-0.10289,-0.08261]; Between **-0.23234**, [-0.25500,-0.20945]; Within **+0.00765**, [+0.00724,+0.00805].

At q=2/3 the Within gain becomes somewhat larger (+0.0205/+0.0210 Arousal; +0.0189/+0.0176 Dominance), while Between degradation becomes much larger (roughly -0.50 to -0.53). q=5 further increases the Between loss. Thus the directionally opposed component profile is stable across all four latent dimensions.

## Audit

- 197,014 utterances and 1,915 speakers.
- 57,450 speaker-moment rows with zero duplicate seed/fold/target/condition/speaker keys.
- 24 delta rows with no missing values.
- 60 FA fit audit rows with no missing values; minimum diagonal residual variance is 0.3495, so no degenerate residual covariance was observed.
- Confidence intervals use 5,000 paired speaker-cluster bootstrap replicates and average the same resampled speakers across the three modeling seeds.

## Interpretation

This experiment strongly supports component attribution but **does not show that this normalization improves standard absolute SER Overall CCC in the present common-representation setup**. Instead, it provides an informative trade-off: removing estimated speaker factors modestly improves within-speaker centered tracking while destroying speaker-level calibration that is useful for predicting the unchanged absolute labels. Overall CCC is dominated by that Between loss.

Together with EXP-20260923-09, the published-method comparison is especially diagnostic: PLDC personalization can selectively improve Between (mu shift), selectively improve Within (sigma shift), or improve both; factor-analysis speaker normalization moves the two components in opposite directions. This supports reporting Overall together with Between and Within rather than interpreting Overall changes as a single capability.

The result should be described as a mechanism reimplementation on the common six-feature Ridge representation, not as a reproduction of Dang et al.'s original ComParE/RVM system.

## Decision

pass; preregistered strong-support criterion is met because the Between-versus-Within profile is stable across all four latent dimensions.
