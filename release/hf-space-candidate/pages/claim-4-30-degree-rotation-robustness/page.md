# Claim 4: 30-degree rotation robustness


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7b726423edef", "created_at": "2026-07-19T17:01:38+00:00", "title": "Claim 4: 30-degree rotation robustness"}
-->
**INCONCLUSIVE.** The exact zero-shot result requires the unreleased trained ImageNet models and validation data.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_26943e1dbdb1", "created_at": "2026-07-19T17:16:50+00:00", "title": "Assessment"}
-->
**Anchored claim.** At a fixed 30-degree ImageNet rotation, nD-RoPE reportedly retains 78.51% top-1 versus 71.34% for mixed RoPE (Table 5, Section 5.2).

**Assessment: INCONCLUSIVE.** This is a zero-shot property of the exact trained ImageNet models. The author release contains no trained checkpoint, and no local ImageNet validation data is available. A random model, generic DeiT checkpoint, or reduced dataset would not test the stated percentage and is therefore not substituted as evidence.

Recovery attempts covered author Git history/releases/branches/issues, the unchanged fork, Hub models/Spaces, and local caches. A faithful next route requires author checkpoints or explicitly authorized full ImageNet training.
