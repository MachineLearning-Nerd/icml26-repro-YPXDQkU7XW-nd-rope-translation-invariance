# Claim 3 EVAL

Final verdict after four routes: **BLOCKED**.

The exact metric cannot yet be verified or falsified because the pinned
release contains no trained ImageNet checkpoint, labels, or per-example
predictions. The released nD model's width 396 differs from the axial and
mixed width 384 models despite the paper's matched-backbone condition. This
confounds the comparison but supplies no contradictory top-1 measurement.
An independent cross-table check finds that nD-RoPE leads RoPE-Mixed by 0.17
points in Table 1 but trails it by 0.18 at zero degrees in Table 5. The paper
does not establish identical checkpoints and preprocessing for those rows, so
the rank reversal remains diagnostic rather than a valid falsification.

The mandatory falsification route considered that rank reversal, an untrained
CPU model test, and the released width confound. Each violates at least one
required assumption or lacks a contradictory full-validation measurement.
No valid counterexample was established.

Unblockers: exact trained model checkpoints, full ImageNet-1K validation
predictions/labels, and a matched architecture or documented justification
for the width-396 nD model.
