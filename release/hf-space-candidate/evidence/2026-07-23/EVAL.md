# EVAL

Fixed cumulative command: `uv run --frozen python repro/src/run_campaign.py`

| Claim | Verdict | Reproduced evidence |
| --- | --- | --- |
| 1 | VERIFIED | f2cae70760806451f5e58be4b7e3dc4d0d856a1e; 7,680 rotary and 192 Fourier trials |
| 2 | VERIFIED | dimensions 2–32, 7,936 symmetry trials, 64 numerical optimizations |
| 3 | BLOCKED | 4 of 4 required routes complete; no faithful full-validation evidence |
| 4 | BLOCKED | 4 of 4 required routes complete; no faithful fixed-checkpoint rotation evidence |
| 5 | FALSIFIED | released 85.97-mIoU path is ShapeNetPart, not ModelNet40 |
| 6 | FALSIFIED | Table 7 contradicts its universal theta=100 statement at the stated 2,048-point training grid |

Independent verifier: `all_checks_pass=True`.
Total runtime: `146.745103` seconds on `macOS-26.5.2-arm64-arm-64bit` with `8` logical CPUs.

Limitations: Claims 3 and 4 have no released trained checkpoints or full
ImageNet prediction evidence. Claim 6's trained metrics were not regenerated;
its verdict follows from a strict internal counterexample that satisfies the
paper's stated Table 7 protocol. No toy or proxy metric is labeled full-scale.
