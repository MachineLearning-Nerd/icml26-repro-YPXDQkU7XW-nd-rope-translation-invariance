# nD-RoPE: claim-by-claim reproduction audit

This repository is the clean-room reproduction and evidence audit for
**[nD-RoPE: A Generalized RoPE for n-Dimensional Position
Embedding](https://arxiv.org/abs/2606.12146)** by Boyang Li, Yulin Wu, Sizhe Xu,
Nuoxian Huang, Zhonghang Yuan, Shangyi Guo, Shu Yang, and Takahiro Yabe.
The paper is accepted to ICML 2026 and proposes a decomposition-free rotary
position embedding based on unified n-dimensional position/frequency vectors
and regular-simplex frequency directions.

Repository: <https://github.com/MachineLearning-Nerd/icml26-nd-rope-translation-invariance>

## What this repository establishes

The publication surface is a scoped audit, not a claim that every paper
experiment was retrained. The current evidence is:

| Paper claim | Status | What produces the status |
| --- | --- | --- |
| C1 — translation-invariant Fourier/Hilbert derivation | `VERIFIED` | Symbolic proof certificates, 7,680 rotary trials, 192 finite Fourier/Parseval cases, official-code parity, and an additive-position negative control. |
| C2 — regular-simplex geometry and economy optimum | `VERIFIED` | Independent constructions for dimensions 2–32, 7,936 permutation trials, 64 economy optimizations, and malformed-simplex controls. |
| C3 — ImageNet in-domain accuracy | `BLOCKED` | Four provenance/architecture/protocol/falsification routes complete; the pinned release has no trained checkpoint or 50,000-image prediction artifact. |
| C4 — 30-degree ImageNet rotation robustness | `BLOCKED` | The exact fixed checkpoints, validation data, and complete rotation preprocessing contract are unavailable; no proxy model is promoted. |
| C5 — cross-modal benchmark results | `FALSIFIED_AS_WRITTEN` | The released 85.97-mIoU path is ShapeNetPart part segmentation, not ModelNet40; SemanticKITTI implementation is absent. |
| C6 — ablations and computational cost | `FALSIFIED_AS_WRITTEN` | At 2,048 points, the paper table has `theta=2` above `theta=100`; exact released image models also show higher attention cost caused by width 396 versus 384, not learned frequency parameters. |

The repository-native publication gate reports `SCOPED_PASS`: all six claim
rows are represented by checked-in contracts, executable producers, raw
outputs, independent checks, and negative controls. The live evaluator snapshot
stored in the evidence bundle is `6/12`, retrieved on 2026-07-23; it is a
historical external result, not a forecast or a new score claim.

## How each claim is produced

The fixed campaign entry point is:

```bash
uv sync --frozen
uv run --frozen python repro/src/run_campaign.py
```

The campaign runs the following evidence paths:

1. `repro/src/run_claim1.py` evaluates translation invariance, relative
   displacement, Parseval, Riesz-kernel preservation, and official-code parity.
   `repro/src/run_symbolic_proofs.py` supplies the universal symbolic
   certificates. Results are in `outputs/claim1/` and
   `outputs/symbolic_proof_certificates.json`.
2. `repro/src/run_claim2.py` independently constructs the regular simplex,
   checks rank/centroid/Gram/tight-frame identities, tests permutation symmetry,
   and minimizes the economy objective. Results are in `outputs/claim2/`.
3. `repro/src/run_claim34.py` executes four fail-closed routes for C3 and C4:
   artifact provenance, architecture/training contract, cross-table protocol,
   and assumption-preserving falsification. Results and route history are in
   `outputs/claim34/` and `.openresearch/artifacts/claim{3,4}/`.
4. `repro/src/run_source_audit.py` inventories the pinned author release and
   reads the actual dataset/metric entrypoints. Its output is
   `outputs/source_audit/source_audit.json`.
5. `repro/src/run_claim6.py` checks the exact table contracts and negative
   controls. `repro/src/run_claim6_flops.py` executes the released 224×224
   image models with two CPU operation counters and a matched-width control.
   Results are in `outputs/claim6/`.
6. `repro/src/verify_results.py` re-reads raw outputs and performs 52
   fail-closed checks. `repro/src/run_verifier_failure_controls.py` mutates
   one decisive value for each claim and confirms six nonzero verifier exits.

No paper-table accuracy is described as independently reproduced when its
checkpoint, predictions, or full protocol were not available. The C3/C4
`BLOCKED` labels are deliberate evidence results, not missing work disguised as
success.

## Branch map

The original `master`/`orx/*` experiment lineage is preserved as descriptive
branches. The final names and roles are:

| Final branch | Role | Outcome |
| --- | --- | --- |
| `baseline/judged-6-of-12` | Frozen judged baseline | C1/C2 verified and C5 falsified under the initial evidence surface. |
| `audit/claim-6-table-contradiction` | Exact C6 table-contract audit | Found the 2,048-point `theta=2` counterexample. |
| `audit/claims-3-4-artifact-provenance` | C3/C4 route 1 | No checkpoint, ImageNet labels, or predictions in the complete release. |
| `audit/claims-3-4-architecture-contract` | C3/C4 route 2 | Width 396/66 versus 384/64 comparison confound identified. |
| `audit/claims-3-4-cross-table` | C3/C4 route 3 | Table rank reversal found, but not a valid falsification without identical protocol evidence. |
| `audit/claims-3-4-falsification` | C3/C4 route 4 | Exact assumption-preserving falsification search remains blocked. |
| `release/cumulative-evidence` | First cumulative release candidate | Combined claim contracts and independent verification. |
| `release/final-approval-candidate` | Pre-publication approval package | Frozen evidence package for review. |
| `audit/post-judge-c6-flops` | Post-judge C6 dynamic profiler | Exact released-model FLOP/parameter attribution. |
| `release/post-judge-c6-dual-profiler` | Independent dual-counter release | Profiler and dispatch-counter agreement. |
| `release/post-judge-publication` | Post-judge publication package | Immutable evaluator bundle and upload manifest. |
| `release/evaluator-visible-verification` | Canonical current verifier | Reader-facing six-claim evidence surface. |
| `release/blind-review-candidate` | Blind-review candidate | Independently traversed candidate package. |
| `main` | Current publication surface | This README, normalized metadata, source pins, and final gate. |

The detailed old-to-new mapping, original tips, and final GitHub tips are in
[`docs/BRANCH_AUDIT.md`](docs/BRANCH_AUDIT.md).

## Source and evidence boundaries

- Paper source and PDF are pinned under `sources/arxiv/`; their SHA-256 values
  are recorded in [`sources.json`](sources.json).
- The author implementation is vendored for audit under `vendor/nD-RoPE/` and
  pinned to [`BoyangL1/nD-RoPE@f2cae707`](https://github.com/BoyangL1/nD-RoPE/tree/f2cae70760806451f5e58be4b7e3dc4d0d856a1e).
- `release/hf-space-candidate/` is retained as an archival snapshot of the
  evaluator-visible package. It contains historical Trackio-rendered pages;
  it is not the canonical local experiment state and is not regenerated by
  the local claim campaign.
- No GPU, paid remote compute, private checkpoint, or private evaluator token
  is included in this repository.

See [`docs/CLAIM_EVIDENCE.md`](docs/CLAIM_EVIDENCE.md) for the claim-to-file
matrix, [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md) for provenance and
limitations, and [`docs/PUBLICATION_GATE.md`](docs/PUBLICATION_GATE.md) for
the reproducibility gate.

## Citation

```bibtex
@inproceedings{li2026ndrope,
  title     = {nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding},
  author    = {Boyang Li and Yulin Wu and Sizhe Xu and Nuoxian Huang and Zhonghang Yuan and Shangyi Guo and Shu Yang and Takahiro Yabe},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  eprint    = {2606.12146},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

## Thank you

Thank you to Boyang Li, Yulin Wu, Sizhe Xu, Nuoxian Huang, Zhonghang Yuan,
Shangyi Guo, Shu Yang, and Takahiro Yabe for releasing the nD-RoPE code and
for making the mathematical and implementation details available for public
study. This audit is intended as a respectful, reproducible companion to the
paper: disagreements are recorded with executable evidence and explicit
limitations rather than presented as judgments about the authors.

## License and attribution

The vendored author implementation retains its upstream license and attribution
files. This repository's scripts and audit documents are maintained by
MachineLearning-Nerd; upstream code and the paper remain the authors' work.
