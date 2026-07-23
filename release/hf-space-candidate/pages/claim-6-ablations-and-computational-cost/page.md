# Claim 6: Ablations and computational cost


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8b4f09cc963f", "created_at": "2026-07-19T17:01:38+00:00", "title": "Claim 6: Ablations and computational cost"}
-->
**PARTIAL; EMPIRICS INCONCLUSIVE.** Theory and reported-table arithmetic are checked, but the ablation checkpoints/configurations are absent.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_dc8dc55dc471", "created_at": "2026-07-19T17:16:51+00:00", "title": "Assessment"}
-->
**Anchored claim.** The paper reports a preferred 6x10 scale/head allocation, preferred 3D base `theta=100`, and negligible added FLOPs (Tables 6–8).

**Assessment: PARTIAL / EMPIRICS INCONCLUSIVE.** Exact paper-table arithmetic gives 6x10 the best mean accuracy (`80.3813`) and theta 100 the best mean mIoU (`79.3643`). Equation 37 independently bounds the 3D base by `207.127`, placing 100 inside and `10^4`/`10^6` outside. Table 8 arithmetic gives +6.07% FLOPs/+5.89% parameters for the widened image model, 0% for video, and about +2% FLOPs for point-cloud vector attention.

The release contains no ablation checkpoints/configurations. It implements 11 scales x 6 heads at width 396, not the paper's stated fixed-384-channel 6x10 layout (which accounts for 360 simplex channels). Therefore the table reductions and theory are checked, but the learned empirical optima are not claimed independently reproduced.
