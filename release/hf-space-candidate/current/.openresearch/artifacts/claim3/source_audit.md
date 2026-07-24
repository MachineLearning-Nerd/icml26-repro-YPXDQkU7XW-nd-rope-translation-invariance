# Claim 3 source audit

The hash-pinned paper reports 81.07, 80.89, and 80.90 in Table 1 and requires
matched ViT-S/DeiT-S architecture and training in Section 5 and Appendix C.
The pinned 154-file official release has no trained checkpoints or ImageNet
predictions. Its released nD constructor is width 396 with six heads
(head dimension 66), while the axial and mixed constructors are width 384
with six heads (head dimension 64).

The width mismatch is a material fairness and reproducibility risk, but it is
not itself a contradictory top-1 measurement.
