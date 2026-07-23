# Claim 3 EVAL

Provisional verdict after routes 1–3: **UNRESOLVED**.

The exact metric cannot yet be verified or falsified because the pinned
release contains no trained ImageNet checkpoint, labels, or per-example
predictions. The released nD model's width 396 differs from the axial and
mixed width 384 models despite the paper's matched-backbone condition. This
confounds the comparison but supplies no contradictory top-1 measurement.
An independent cross-table check finds that nD-RoPE leads RoPE-Mixed by 0.17
points in Table 1 but trails it by 0.18 at zero degrees in Table 5. The paper
does not establish identical checkpoints and preprocessing for those rows, so
the rank reversal remains diagnostic rather than a valid falsification.
