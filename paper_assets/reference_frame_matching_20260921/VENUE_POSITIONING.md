# Venue Positioning

## Primary recommendation: Interspeech 2027

Why:
- the core contribution is speech/prosody/SER rather than language understanding;
- the main experiments are acoustic and affective;
- Interspeech has an established SER community and directly adjacent recent work on speaker-aware
  SER, naturalistic emotional attribute prediction, prosody, and speaker adaptation;
- the current experimental package fits a focused conference paper well.

Current public deadline:
- Paper submission: 9 February 2027.

Recommended version:
- center the paper on the reference-frame mechanism;
- keep WavLM analysis compact or in appendix/supplement;
- use the semantically matched MSP/IEMOCAP Pitch+Rate intervention as the main result;
- include center permutation and K-shot as mechanism/deployment validation.

## Strong journal option: IEEE Transactions on Affective Computing

Why:
- the work is primarily about principles of affective sensing rather than a new architecture;
- the controlled target intervention and mechanism analysis are better suited to a longer paper;
- negative/sensitivity findings materially strengthen a journal story;
- there is room for categorical state/trait experiments, VAD mechanism, K-shot, SSL analysis,
  annotation reliability, and session-level limitations in one coherent submission.

Recommended additions for a journal version:
- a third genuine continuous-affect corpus if one becomes available;
- stronger session/domain random-effects analysis;
- possibly a learned reference estimator rather than median-only K-shot reference;
- human/perceptual validation of relative versus absolute interpretation.

## Secondary journal option: IEEE/ACM TASLP

Good fit if the paper is framed more as:
- speech representation / prosodic modeling;
- normalization and information decomposition;
- reference estimation for downstream speech tasks.

TAFFC is the cleaner fit if the affective-theory angle remains central.

## COLING 2027

Current ARR submission deadline:
- 12 October 2026.

Fit:
- possible, but weaker in the current form;
- the main contribution is acoustic/prosodic rather than computational-linguistic;
- a COLING version would need a stronger language-facing story, e.g. interaction with transcripts,
  lexical semantics, spoken-language models, or language-conditioned affect.

Recommendation:
Do not distort the current paper merely to make it look like NLP. The present evidence is stronger
as a speech/affective computing paper.

## ICASSP 2027

Current official full-paper deadline is listed as 23 September 2026.

Scientific fit:
- strong.

Practical fit:
- poor for this new paper at the current date because the deadline is immediate and the project
  would have to be compressed aggressively.
- A rushed ICASSP version would likely omit the mechanism depth that makes the paper distinctive.

Recommendation:
Do not sacrifice the target-reference story to meet the immediate deadline.
