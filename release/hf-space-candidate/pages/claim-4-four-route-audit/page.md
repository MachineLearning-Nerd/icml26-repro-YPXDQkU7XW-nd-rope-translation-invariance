# Claim 4: four-route audit

**Claim contract.** With fixed ImageNet models and no fine-tuning, at 30°
nD-RoPE scores 78.51% top-1 while RoPE-Mixed scores 71.34%.

**Verdict: BLOCKED. Confidence: LOW.**

## Completed routes

1. **Artifact and data provenance.** No trained checkpoint, ImageNet
   full-validation predictions, or labels occur in the pinned release.
2. **Protocol audit.** No rotation evaluator is released. The paper specifies
   resize to 256, center crop to 224, and rotation, but omits resize and
   rotation interpolation, fill, and antialias settings.
3. **Cross-table check.** The zero-degree row reverses the Table 1 nD/Mixed
   rank. The paper does not establish identical checkpoint identities and
   preprocessing, so this is not an assumption-preserving counterexample.
4. **Mandatory exact falsification search.** A random-model run would violate
   the 400-epoch trained-model assumption. The source confound contains no
   contradictory top-1 measurement. No valid counterexample remains.

## Unblockers

The exact fixed checkpoints used in Table 5, ImageNet-1K validation access,
and complete resize/rotation interpolation, fill, and antialias settings.

Raw route records and negative controls are in
`evidence/2026-07-23/claim34_report.json` and
`evidence/2026-07-23/claim34_negative_controls.json`.
