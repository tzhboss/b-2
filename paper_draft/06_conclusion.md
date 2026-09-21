# 6. Conclusion

We studied speaker normalization as a reference-frame choice rather than a universally beneficial
preprocessing operation.

Across categorical state-versus-trait tasks, continuous VAD decomposition, a controlled
between-speaker target intervention, speaker-center permutation, and label-free enrollment, the
results support one consistent principle: the usefulness of stable speaker information depends on
the reference frame of the target.

When the target represents within-speaker affect deviation, speaker-relative prosody is consistently
advantageous for Arousal and Dominance. As stable between-speaker target structure is introduced,
Relative becomes less favorable and speaker baseline information can become useful. This coupling
replicates across MSP-Podcast and IEMOCAP using semantically matched Pitch + Speaking Rate
features and both linear and quadratic models.

The practical implication is not that speaker normalization should always be applied, nor that it
should always be avoided. Instead, SER systems should make an explicit choice about whether the
prediction target is population-relative or person-relative, and construct their acoustic
representation accordingly.

In short:

**Representation reference frame should match target reference frame.**
