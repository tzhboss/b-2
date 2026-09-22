# CURRENT_STATE.md

## Phase

Core reference-frame experiments and numerical robustness checks are complete.

## Strongest current evidence

1. MSP explicit Pitch+Rate target intervention:
   strong speaker-cluster slope confirmation under linear and quadratic Ridge.
2. IEMOCAP explicit Pitch+Rate:
   negative point slopes but wide speaker-cluster slope CIs with only 10 speakers; pure
   within-speaker Relative endpoint is robust.
3. MSP extended attributes:
   F0 variability, pause ratio, voiced ratio, and their combination all show target-reference
   coupling.
4. IEMOCAP WavLM embedding intervention:
   layer 12 and layer 24 Arousal/Dominance slopes all significantly negative under speaker-cluster
   inference; pure within-speaker endpoints all significantly favor Relative.
5. WavLM numerical solver sensitivity:
   LSQR reproduces the default-solver effects to within 1.7e-6.

## Main paper interpretation

The target-reference mechanism is no longer confined to handcrafted scalar prosody. It appears in
both interpretable prosodic attributes and high-dimensional learned speech representations.

## Remaining high-value gap

External breadth remains the main limitation. A third licensed continuous-affect corpus or
MSP-Conversation contextual labels would be valuable if access becomes available. Standard
voice-quality features are secondary to this gap.

## Next legal step

Update the manuscript evidence hierarchy, figures, and claims to reflect speaker-cluster inference,
extended attributes, WavLM direct intervention, and solver stability.
