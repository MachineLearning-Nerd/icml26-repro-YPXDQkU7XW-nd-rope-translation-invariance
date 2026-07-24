# Claim 3 route ledger

Every route runs through the inherited command:
`uv run --frozen python repro/src/run_campaign.py`.

1. **Artifact provenance.** Interpreted the numeric claim as requiring exact
   trained states and 50,000 labeled predictions. The pinned 154-file release
   contains neither. Result: unresolved; absence is not falsification.
2. **Architecture/training contract.** Interpreted “same backbone” literally
   and parsed the official constructors. nD uses width 396/head dimension 66;
   axial and mixed use 384/64. Result: material confound, but no contradictory
   top-1 measurement.
3. **Cross-table check.** Interpreted Table 5's zero-degree row as a possible
   independent in-domain cross-check. The nD-versus-Mixed rank reverses, but
   checkpoint and preprocessing identity are not established. Result:
   diagnostic, not falsification.
4. **Exact falsification search.** Restated ImageNet-1K, 400-epoch trained
   ViT-S models, matched protocol, and full 224 evaluation as mandatory.
   Cross-table, random-model, and width-confound candidates each violate an
   assumption or lack contradictory predictions. Result: no valid
   counterexample; **BLOCKED**.

The raw route report and negative controls are generated under
`.openresearch/artifacts/claim3/raw/`; the independent checker is written to
`.openresearch/artifacts/claim3/independent_checker.json`.
