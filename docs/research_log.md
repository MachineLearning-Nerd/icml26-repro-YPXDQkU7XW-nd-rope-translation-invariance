# Research log

## 2026-08-14 audit

- Confirmed the repository's current GitHub metadata: public, default branch
  `master`, description identifying OpenReview `YPXDQkU7XW`, and 13 `orx/*`
  experiment branches.
- Confirmed the paper identity and author list against arXiv `2606.12146v1`.
- Confirmed the official implementation commit
  `f2cae70760806451f5e58be4b7e3dc4d0d856a1e` through the upstream GitHub API.
- Read the local claim contracts, raw outputs, source inventory, independent
  verifier, evaluator package, and release gate before editing publication
  docs.
- Confirmed the existing evidence gate reports 52/52 checks and six mutation
  cases passing.
- Downloaded and SHA-256 pinned the arXiv source archive and PDF under
  `sources/arxiv/`.

## Interpretation policy

Paper-reported metrics are separated from independently produced evidence.
Absence of a checkpoint is recorded as a reproducibility boundary, not as a
numeric counterexample. A falsification is reported only where an executable
source or exact released model produces a contradiction to the written
contract.
