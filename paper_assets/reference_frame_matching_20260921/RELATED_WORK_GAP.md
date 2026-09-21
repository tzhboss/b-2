# Related-Work Gap: Prosodic Reference Frames

## Closest prior lines

### Speaker normalization for emotion recognition

Prior SER work has long treated speaker-dependent acoustic variation as a nuisance to compensate.

Representative examples:
- Busso et al., Iterative Feature Normalization Scheme for Automatic Emotion Detection from Speech
  (IEEE Transactions on Affective Computing, 2013): estimates speaker normalization statistics while
  aiming to preserve inter-emotional variability.
- Mariooryad/Busso line on compensating speaker and lexical variability
  (Speech Communication, 2014): factorizes speaker, lexical, and emotional dependencies and applies
  whitening-based normalization.
- Gat et al., Speaker Normalization for Self-Supervised Speech Emotion Recognition
  (ICASSP 2022): adversarially suppresses speaker characteristics to improve SER generalization.

These works motivate speaker normalization, but their dominant framing is:
speaker information -> nuisance / shortcut -> remove or compensate.

## Between-speaker versus within-speaker acoustic variability

Phonetic and sociophonetic studies explicitly distinguish stable between-speaker baselines from
within-speaker variation. This is well established for speaking rate, F0, rhythm, and voice-quality
measures.

These works motivate the decomposition:
x_su = speaker baseline + within-speaker deviation.

They generally do not ask which component should be preserved for a downstream affect target.

## Speaker dependence of affect labels

Prior dimensional-emotion work has observed that speaker dependence affects VAD prediction and
speaker-independent generalization. For example, Sridhar et al. (Interspeech 2018) reported
speaker-dependent traits in valence prediction and linked them to regularization/generalization.

This establishes that affective targets can contain speaker-linked structure, but it does not
directly test how the target's between-speaker component should determine the acoustic
normalization/reference frame.

## Distinct contribution supported by current experiments

The current evidence is not simply that speaker normalization helps.

The supported mechanism is:

1. Decompose acoustic prosody into stable speaker center and within-speaker deviation.
2. Decompose the target into between-speaker and within-speaker components.
3. Manipulate the amount of between-speaker target structure while holding inputs/model fixed.
4. Observe a systematic change in Relative-versus-Absolute utility.
5. Show that the effect survives two corpora, aligned Pitch+Rate features, and linear/quadratic
   model families.
6. Show by speaker-center permutation that correct center identity matters only when the target
   contains speaker-level structure.
7. Show deployment feasibility through label-free K-shot speaker reference estimation.

The paper therefore should not claim novelty in speaker normalization itself.

The stronger gap statement is:

> Existing work primarily asks how to suppress speaker variability for robust emotion recognition.
> We ask when speaker variability is nuisance versus task-relevant information, and show that the
> answer is governed by the reference frame of the target.

## Claims to avoid

- "Speaker normalization is new."
- "Relative prosody is universally superior."
- "Speaker information is always nuisance for emotion recognition."
- "Relative prosody removes speaker identity."
- "IEMOCAP loudness independently validates the mechanism."

## Safer novelty formulation

A defensible formulation is:

> We formulate prosodic normalization as a reference-frame choice rather than a universally
> beneficial invariance operation. By explicitly manipulating the between-speaker component of
> dimensional affect targets, we show that the preferred prosodic reference frame changes
> systematically with the target reference frame.

This wording still requires a full final literature review before submission.
