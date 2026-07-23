# Claim 3: four-route audit

**Claim contract.** On ImageNet-1K with a ViT-S backbone at 224×224, nD-RoPE
achieves 81.07% top-1 and exceeds Axial RoPE at 80.89% and RoPE-Mixed at
80.90%.

**Verdict: BLOCKED. Confidence: LOW.**

## Completed routes

1. **Artifact and data provenance.** The complete 154-file author release at
   `f2cae70760806451f5e58be4b7e3dc4d0d856a1e` has no trained checkpoint,
   ImageNet labels, or per-example predictions. A fail-closed gate rejects
   partial predictions, copied table values, and missing checkpoint hashes.
2. **Architecture and training contract.** The released nD constructor is
   width 396/head dimension 66. Axial and Mixed are width 384/head dimension
   64. This is a comparison confound, not a contradictory accuracy value.
3. **Independent cross-table check.** Table 1 gives nD minus Mixed as +0.17
   points, while Table 5 at 0° gives −0.18. Checkpoint and preprocessing
   identity are not established, so the reversal is diagnostic only.
4. **Mandatory exact falsification search.** The rank reversal, randomly
   initialized CPU models, and the width confound were each tested as
   candidates. None both satisfies the trained ImageNet assumptions and
   supplies contradictory full-validation predictions.

## Unblockers

The exact trained nD-RoPE, Axial, and Mixed checkpoints; ImageNet-1K
validation labels and per-example predictions; and a matched architecture or
documented justification for width 396.

Raw route records and negative controls are in
`evidence/2026-07-23/claim34_report.json` and
`evidence/2026-07-23/claim34_negative_controls.json`.
