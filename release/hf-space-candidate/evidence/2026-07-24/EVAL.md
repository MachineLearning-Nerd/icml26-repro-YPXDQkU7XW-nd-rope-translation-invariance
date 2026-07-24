# EVAL

Fixed cumulative command: `uv run --frozen python repro/src/run_campaign.py`

| Claim | Verdict | Reproduced evidence |
| --- | --- | --- |
| 1 | VERIFIED | f2cae70760806451f5e58be4b7e3dc4d0d856a1e; 7,680 rotary and 192 Fourier trials |
| 2 | VERIFIED | dimensions 2–32, 7,936 symmetry trials, 64 numerical optimizations |
| 3 | BLOCKED | 4 of 4 required routes complete; no faithful full-validation evidence |
| 4 | BLOCKED | 4 of 4 required routes complete; no faithful fixed-checkpoint rotation evidence |
| 5 | FALSIFIED | released 85.97-mIoU path is ShapeNetPart, not ModelNet40 |
| 6 | FALSIFIED | exact 224x224 models dynamically profiled; 5.69% extra attention MACs are caused by width 396 vs 384, while frequency directions contain zero trainable parameters |

Independent verifier: `all_checks_pass=True`.
Verifier mutation controls: all six claims rejected corrupted evidence with
nonzero exits.
Total runtime: `119.595468` seconds on `macOS-26.5.2-arm64-arm-64bit` with `8` logical CPUs.

Limitations: Claims 3 and 4 have no released trained checkpoints or full
ImageNet prediction evidence. Claim 6's Table 6/7 trained metrics were not
regenerated. Its decisive post-judge route instead executes and profiles the
exact released Table 8 image architectures at 224x224, with a matched-width
attribution control and an independent symbolic checker. No toy or proxy
metric is labeled full-scale.
