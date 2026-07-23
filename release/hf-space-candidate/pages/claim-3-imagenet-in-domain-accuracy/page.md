# Claim 3: ImageNet in-domain accuracy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_98ffe58be1ac", "created_at": "2026-07-19T17:01:38+00:00", "title": "Claim 3: ImageNet in-domain accuracy"}
-->
**INCONCLUSIVE.** No checkpoint is released, and the public nD model is widened relative to its baselines; no proxy is substituted.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7b431e46bfbb", "created_at": "2026-07-19T17:16:50+00:00", "title": "Assessment"}
-->
**Anchored claim.** On ImageNet-1K at 224x224, nD-RoPE reportedly reaches 81.07% top-1 versus 80.89% axial and 80.90% mixed RoPE (Table 1).

**Assessment: INCONCLUSIVE.** No checkpoint is present in the full 154-file author release, its two-commit history, releases, Hub search, unchanged fork, or local cache. The released comparison also has a material backbone confound: nD-RoPE uses width 396 / head dimension 66, while the released axial and mixed baselines use width 384 / head dimension 64. The official phase layout rejects head dimension 64 because a 2D simplex scale consumes six channels. Table 8 correspondingly reports 23.36M versus 22.06M parameters.

The exact accuracies were not rerun, and the source mismatch alone is not presented as proof that the numbers are false. Full replication needs the trained checkpoints or authorized 400-epoch ImageNet training.
