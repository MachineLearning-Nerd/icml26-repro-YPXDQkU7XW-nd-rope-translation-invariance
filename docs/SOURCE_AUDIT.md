# Source and provenance audit

## Paper

The paper is *nD-RoPE: A Generalized RoPE for n-Dimensional Position
Embedding*, arXiv `2606.12146v1`, OpenReview `YPXDQkU7XW`. The exact arXiv
source archive and PDF are stored under `sources/arxiv/`; their retrieval URLs
and SHA-256 values are in [`sources.json`](../sources.json). The checked-in
`.openresearch/artifacts/source/paper_source.json` preserves the HTML anchor
map used by the claim contracts.

## Official implementation

The author repository is pinned to
[`BoyangL1/nD-RoPE@f2cae70760806451f5e58be4b7e3dc4d0d856a1e`](https://github.com/BoyangL1/nD-RoPE/commit/f2cae70760806451f5e58be4b7e3dc4d0d856a1e).
The complete 154-file release is vendored under `vendor/nD-RoPE/`, with
per-file hashes in `vendor/nD-RoPE/MANIFEST.sha256`. The audit confirms:

- no checkpoint file is present in the pinned release;
- the nD-RoPE image constructor uses width 396, six heads, and head dimension
  66, while the released axial/mixed baselines use width 384 and head
  dimension 64;
- the 85.97-mIoU point-cloud path uses `PartNormalDataset`, 16 ShapeNetPart
  categories, and 50 part labels;
- the separate ModelNet loader is used for classification, not that mIoU
  segmentation result; and
- no SemanticKITTI implementation is present.

These observations are recorded in
`outputs/source_audit/source_audit.json` and are independently asserted by
`repro/src/verify_results.py`.

## Evidence boundaries

The following are intentionally not claimed as full reproduction:

- C3 needs exact trained ImageNet checkpoints, labels, per-example predictions,
  and a matched evaluation protocol.
- C4 additionally needs the exact fixed-model rotation evaluator and its
  interpolation, fill, resize, and antialias settings.
- C6 table arithmetic and exact released-model profiling do not regenerate the
  paper's trained ablation metrics.

No reduced dataset, randomly initialized model, generic pretrained checkpoint,
or subjective “negligible cost” interpretation is substituted for a missing
paper experiment.
