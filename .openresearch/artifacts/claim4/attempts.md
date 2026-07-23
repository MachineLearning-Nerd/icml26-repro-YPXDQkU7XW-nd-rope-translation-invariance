# Claim 4 route ledger

Every route runs through the inherited command:
`uv run --frozen python repro/src/run_campaign.py`.

1. **Artifact provenance.** Required the exact fixed checkpoints and 50,000
   labeled rotated predictions. Neither is released.
2. **Released evaluation path.** Searched the pinned code for the exact
   rotation evaluator. None exists; this does not contradict the paper metric.
3. **Protocol/cross-table check.** Audited the zero-degree row and transform
   description. Interpolation, fill, and antialias are unspecified, and the
   zero-degree checkpoint identity is not established.
4. **Exact falsification search.** Restated the fixed trained models, full
   ImageNet validation set, resize-rotate-crop order, 30 degrees, and no
   fine-tuning as mandatory. Random-model, width-confound, and zero-degree
   candidates fail those assumptions. Result: no valid counterexample;
   **BLOCKED**.

The raw route report and negative controls are generated under
`.openresearch/artifacts/claim4/raw/`; the independent checker is written to
`.openresearch/artifacts/claim4/independent_checker.json`.
