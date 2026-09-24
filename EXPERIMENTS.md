# EXPERIMENTS.md

Human-readable derived index only. The authoritative record is each `experiments/<ID>/experiment.yaml`; if this file conflicts with a YAML record, the YAML wins.

Current inventory: **65 registered experiments; 64 completed.**

| ID | Title | Execution | Validity | Decision | Protocol | Parent |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-20260920-01 | Absolute vs speaker-relative prosody across affective-state and speaker-trait tasks | completed | invalid | inconclusive | prosody_reference_frame_v1 | — |
| EXP-20260920-02 | Absolute vs speaker-relative prosody — corrected controlled pilot | completed | valid | pass | prosody_reference_frame_v2 | EXP-20260920-01 |
| EXP-20260920-03 | Frozen WavLM-large plus absolute versus speaker-relative prosody | completed | valid | inconclusive | prosody_reference_frame_wavlm_v1 | EXP-20260920-02 |
| EXP-20260920-04 | Layer-wise absolute and speaker-relative pitch structure in WavLM-large | completed | valid | inconclusive | wavlm_layerwise_pitch_v1 | EXP-20260920-03 |
| EXP-20260920-05 | Label-free K-shot relative pitch for unseen speakers | completed | valid | mixed | unseen_speaker_kshot_pitch_v1 | EXP-20260920-02, EXP-20260920-04 |
| EXP-20260920-06 | What speaker reference does label-free K-shot pitch estimate? | completed | valid | mixed | reference_target_diagnostic_v1 | EXP-20260920-05 |
| EXP-20260920-07 | Natural-corpus validation of speaker-relative pitch | completed | valid | inconclusive | natural_corpus_reference_v1 | EXP-20260920-05, EXP-20260920-06 |
| EXP-20260920-08 | Why does MSP prefer absolute pitch? | completed | valid | pass | pitch_baseline_decomposition_v1 | EXP-20260920-07 |
| EXP-20260920-09 | Which emotions prefer absolute or relative pitch? | completed | valid | pass | per_emotion_conflict_v1 | EXP-20260920-08 |
| EXP-20260920-10 | Do loudness and speaking rate show the same reference-frame heterogeneity as pitch? | completed | valid | mixed | multi_attribute_emotion_conflict_v1 | EXP-20260920-09 |
| EXP-20260920-11 | Can simple statistics predict which prosodic reference frame will work better? | completed | valid | mixed | reference_preference_predictor_v1 | EXP-20260920-09, EXP-20260920-10 |
| EXP-20260920-12 | Can held-out-corpus statistics route each emotion to the better reference frame? | completed | valid | mixed | reference_frame_router_v1 | EXP-20260920-11 |
| EXP-20260920-13 | Can nested confidence-aware routing improve corpus robustness? | completed | valid | reject | nested_confidence_router_v1 | EXP-20260920-12 |
| EXP-20260920-14 | Are reference-frame effects robust to nonlinear classifiers? | completed | valid | inconclusive | classifier_family_robustness_v1 | EXP-20260920-10 |
| EXP-20260920-15 | Which reference-frame effects depend on classifier family? | completed | valid | inconclusive | per_emotion_classifier_interaction_v1 | EXP-20260920-14 |
| EXP-20260920-16 | Absolute versus speaker-relative prosody for continuous MSP VAD | completed | valid | mixed | msp_vad_reference_frame_v1 | EXP-20260920-10, EXP-20260920-15 |
| EXP-20260920-17 | Does VAD itself contain speaker-level and within-speaker components? | completed | valid | pass | msp_vad_target_decomposition_v1 | EXP-20260920-16 |
| EXP-20260920-18 | Do VAD reference-frame effects survive low annotator disagreement? | completed | valid | mixed | msp_vad_disagreement_v1 | EXP-20260920-16, EXP-20260920-17 |
| EXP-20260920-19 | Can unlabeled K-shot speaker audio recover raw VAD baseline utility? | completed | valid | mixed | msp_vad_kshot_v1 | EXP-20260920-16, EXP-20260920-17, EXP-20260920-18 |
| EXP-20260920-20 | Is K=10 near the unlabeled-reference plateau for raw VAD? | completed | invalid | inconclusive | msp_vad_kshot_saturation_v1 | EXP-20260920-19 |
| EXP-20260920-21 | Does MSP VAD K-shot performance really saturate by K=20? | completed | valid | mixed | msp_vad_kshot_fixedpool_v1 | EXP-20260920-20 |
| EXP-20260921-01 | Immutable rerun of K-shot raw-VAD saturation | completed | valid | mixed | msp_vad_kshot_saturation_v2 | EXP-20260920-19, EXP-20260920-20 |
| EXP-20260921-02 | Independent IEMOCAP validation of VAD reference-frame matching | completed | valid | mixed | iemocap_vad_reference_matching_v1 | EXP-20260920-17, EXP-20260921-01 |
| EXP-20260921-03 | Does reference-frame utility track an intervention on target reference frame? | completed | valid | mixed | vad_target_reference_intervention_v1 | EXP-20260920-17, EXP-20260921-02 |
| EXP-20260921-04 | Is VAD target-reference coupling robust to nonlinear feature access? | completed | valid | pass | vad_target_reference_model_robustness_v1 | EXP-20260921-03 |
| EXP-20260921-05 | Which prosodic attributes carry target-reference coupling? | completed | valid | pass | vad_target_reference_attribute_v1 | EXP-20260921-03, EXP-20260921-04 |
| EXP-20260921-06 | Does Hybrid VAD utility require the correct speaker center? | completed | valid | pass | vad_speaker_center_permutation_v1 | EXP-20260920-17, EXP-20260921-03, EXP-20260921-05 |
| EXP-20260921-07 | Does target-reference coupling survive K=20 unlabeled speaker enrollment? | completed | valid | mixed | vad_kshot_target_reference_intervention_v1 | EXP-20260920-21, EXP-20260921-03, EXP-20260921-06 |
| EXP-20260921-08 | Does better unlabeled reference estimation recover the IEMOCAP target-reference slope? | completed | valid | mixed | iemocap_kshot_reference_quality_curve_v1 | EXP-20260921-02, EXP-20260921-03, EXP-20260921-07 |
| EXP-20260921-09 | Does IEMOCAP target-reference coupling survive replacing relative_db with RMS dB? | completed | valid | reject | iemocap_loudness_semantic_robustness_v1 | EXP-20260921-03, EXP-20260921-05, EXP-20260921-08 |
| EXP-20260921-10 | Does IEMOCAP target-reference coupling survive after removing ambiguous loudness? | completed | valid | pass | iemocap_pitch_rate_reference_matching_v1 | EXP-20260921-04, EXP-20260921-05, EXP-20260921-09 |
| EXP-20260921-11 | Does speaker-relative prosody suppress speaker identity beyond gender? | completed | valid | mixed | speaker_identity_decodability_v1 | EXP-20260920-02, EXP-20260920-04 |
| EXP-20260921-12 | Semantically matched cross-corpus pitch+rate reference-frame validation | completed | valid | pass | crosscorpus_pitch_rate_reference_matching_v1 | EXP-20260921-04, EXP-20260921-09, EXP-20260921-10 |
| EXP-20260921-13 | Does IEMOCAP target-reference coupling survive leave-one-session-out evaluation? | completed | invalid | inconclusive | iemocap_session_disjoint_reference_matching_v1 | EXP-20260921-10, EXP-20260921-12 |
| EXP-20260921-14 | Does target-reference coupling extend to pitch variability and temporal prosody? | completed | valid | pass | msp_extended_prosody_reference_intervention_v1 | EXP-20260921-05, EXP-20260921-12 |
| EXP-20260921-15 | Speaker-cluster inference for the main target-reference effect | completed | valid | pass | pitch_rate_speaker_cluster_inference_v1 | EXP-20260921-12, EXP-20260921-13 |
| EXP-20260921-16 | Does target-reference matching survive in frozen WavLM embedding space? | completed | valid | pass | iemocap_wavlm_reference_intervention_v1 | EXP-20260920-04, EXP-20260921-12, EXP-20260921-15 |
| EXP-20260921-17 | Is the WavLM target-reference result stable to Ridge solver choice? | completed | valid | pass | iemocap_wavlm_solver_stability_v1 | EXP-20260921-16 |
| EXP-20260921-18 | Can unlabeled K-shot enrollment recover WavLM reference-frame matching? | completed | valid | pass | iemocap_wavlm_kshot_reference_v1 | EXP-20260921-16, EXP-20260921-17 |
| EXP-20260922-01 | Full interpretable prosody target-reference matching | completed | valid | pass | msp_full_interpretable_prosody_cluster_v1 | EXP-20260921-14, EXP-20260921-15 |
| EXP-20260922-02 | Can unlabeled K-shot enrollment recover the full interpretable-prosody Hybrid benefit? | completed | valid | pass | msp_full_interpretable_kshot_fixedpool_v1 | EXP-20260920-21, EXP-20260922-01 |
| EXP-20260923-01 | Standard absolute SER Overall/Between/Within diagnostic decomposition | completed | valid | mixed | standard_ser_component_diagnostics_v1 | EXP-20260920-17, EXP-20260922-01 |
| EXP-20260923-02 | Cross-model Overall/Between/Within profiles on IEMOCAP | completed | valid | mixed | iemocap_crossmodel_component_profiles_v1 | EXP-20260921-15, EXP-20260921-16, EXP-20260923-01 |
| EXP-20260923-03 | emotion2vec standard absolute VAD component profile on IEMOCAP | completed | valid | weak_for_rank_reversal | iemocap_emotion2vec_component_profile_v1 | EXP-20260923-02 |
| EXP-20260923-04 | Corrected MSP center-permutation randomization audit | completed | valid | pass | msp_center_permutation_seedfix_v1 | EXP-20260921-06, EXP-20260923-01 |
| EXP-20260923-05 | Corrected full-prosody K-shot recovery with speaker-cluster inference | completed | valid | pass | msp_full_interpretable_kshot_speakercluster_v2 | EXP-20260922-02, EXP-20260923-01 |
| EXP-20260923-06 | WavLM target-reference matching under IEMOCAP session-disjoint retraining | completed | valid | pass | iemocap_wavlm_session_loso_reference_v1 | EXP-20260921-16, EXP-20260921-17 |
| EXP-20260923-07 | Same-budget raw-x speaker-center identity control | completed | valid | mixed | msp_rawx_center_identity_control_v1 | EXP-20260921-06, EXP-20260923-04 |
| EXP-20260923-08 | LSQR sensitivity for WavLM session-disjoint target-reference matching | completed | valid | pass | iemocap_wavlm_session_loso_reference_v1 | EXP-20260923-06, EXP-20260921-17 |
| EXP-20260923-09 | Published PLDC mechanism under Overall/Between/Within attribution | completed | valid | pass | msp_pldc_component_attribution_v1 | EXP-20260923-01 |
| EXP-20260923-10 | Published continuous-SER factor-analysis normalization under component attribution | completed | valid | pass | msp_dang2016_fa_normalization_attribution_v1 | EXP-20260923-01 |
| EXP-20260924-01 | Paired cross-model Overall/Between/Within attribution on IEMOCAP | completed | valid | mixed | iemocap_paired_crossmodel_component_attribution_v1 | EXP-20260923-02, EXP-20260923-03 |
| EXP-20260924-02 | HuBERT and wav2vec2 extension of IEMOCAP component profiles | completed | valid | pass | iemocap_hubert_wav2vec2_component_extension_v1 | EXP-20260924-01 |
| EXP-20260924-03 | Absolute/Relative/Hybrid component intervention across frozen speech backbones | completed | valid | mixed | iemocap_absolute_relative_hybrid_component_v1 | EXP-20260924-02, EXP-20260923-01 |
| EXP-20260924-04 | Ridge-alpha sensitivity of Absolute/Relative/Hybrid component effects | completed | valid | pass | iemocap_arh_alpha_sensitivity_v1 | EXP-20260924-03 |

| EXP-20260924-05 | Session-disjoint cross-model Overall/Between/Within attribution on IEMOCAP | completed | valid | pass | iemocap_session_loso_crossmodel_component_v1 | EXP-20260924-02, EXP-20260923-06 |

| EXP-20260924-06 | LSQR sensitivity for session-disjoint cross-model component attribution | completed | valid | mixed | iemocap_session_loso_crossmodel_lsqr_v1 | EXP-20260924-05, EXP-20260923-08 |

| EXP-20260924-07 | Session-disjoint Absolute/Relative/Hybrid component intervention across frozen speech backbones | completed | valid | mixed | iemocap_session_loso_arh_component_v1 | EXP-20260924-03, EXP-20260924-05 |

| EXP-20260924-08 | LSQR sensitivity for session-disjoint Absolute/Relative/Hybrid intervention | completed | valid | mixed | iemocap_session_loso_arh_lsqr_v1 | EXP-20260924-07, EXP-20260924-06 |

| EXP-20260924-09 | High-precision LSQR sensitivity for session-disjoint Absolute/Relative/Hybrid intervention | completed | valid | pass | iemocap_session_loso_arh_lsqr_highprecision_v1 | EXP-20260924-07, EXP-20260924-08, EXP-20260923-08 |

| EXP-20260924-10 | Nested session-LOSO component-aware recomposition versus ensemble and PLDC baselines | completed | valid | mixed | iemocap_nested_component_recomposition_benchmark_v1 | EXP-20260924-05, EXP-20260923-09, EXP-20260923-05 |

| EXP-20260924-11 | Component-balanced maximin ensemble under nested session LOSO | completed | valid | pass | iemocap_component_balanced_ensemble_v1 | EXP-20260924-10, EXP-20260924-05 |

| EXP-20260924-12 | PCM-protocol fixed-split comparison for component-balanced absolute SER | completed | valid | pass | iemocap_pcm_fixed_split_comparison_v1 | EXP-20260924-11 |

| EXP-20260924-13 | LSQR sensitivity for component-balanced maximin ensemble | completed | valid | pass | iemocap_component_balanced_lsqr_sensitivity_v1 | EXP-20260924-11, EXP-20260924-06 |

| EXP-20260924-14 | Paired component-balanced maximin versus PLDC and ensemble controls | registered | pending | pending | iemocap_component_balanced_vs_pldc_v1 | EXP-20260924-10, EXP-20260924-11, EXP-20260924-13 |
