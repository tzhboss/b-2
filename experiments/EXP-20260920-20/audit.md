# Experiment Audit — EXP-20260920-20

## Runtime provenance failure

The completed artifact directory cannot be treated as the preregistered EXP-20 run.

The tracked and preregistered script at HEAD has SHA256:
d605fe601fd5aa27fbfc3d43f61535d44a71c018f96be77535017ec75151b126

However, the produced runtime metadata and output schema correspond to a different implementation:
- runtime metadata reports common_min_support: 51 and single nested random order per speaker;
- runtime artifacts include audit_rows.csv, center_error_by_speaker.csv, and paired_deltas.csv;
- the preregistered tracked script instead specifies min_support_exclusive and outputs k_deltas.csv.

The Git worktree after execution is clean and the tracked script matches the preregistered commit,
so the mismatch occurred at runtime/output provenance level, likely due concurrent/NFS path reuse.

## Verdict

- Execution: completed.
- Validity: invalid.
- Decision: inconclusive.
- No numerical result from this artifact directory may be used as EXP-20 evidence.
- Artifacts are preserved for audit and are not promoted.
