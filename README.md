# nD-RoPE reproduction and source audit

Reproduction of **nD-RoPE: A Generalized RoPE for n-Dimensional Position
Embedding** ([OpenReview YPXDQkU7XW](https://openreview.net/forum?id=YPXDQkU7XW),
[arXiv 2606.12146](https://arxiv.org/abs/2606.12146)). The author source is
pinned to [`BoyangL1/nD-RoPE@f2cae707`](https://github.com/BoyangL1/nD-RoPE/tree/f2cae70760806451f5e58be4b7e3dc4d0d856a1e).

## Outcome

| Anchored claim | Assessment | Direct evidence |
| --- | --- | --- |
| C1 — Fourier/Hilbert derivation | **Verified** | 7,680 rotary identities, 192 finite Parseval/Riesz cases, official-code parity, additive-PE negative control |
| C2 — simplex geometry and optimal ratio | **Verified** | dimensions 2–32, 7,936 permutation checks, 64 independent optimizations, 15 malformed-geometry controls |
| C3 — ImageNet 81.07% | **Inconclusive** | no checkpoint; released nD model is width 396 while released baselines are width 384 |
| C4 — 30-degree accuracy 78.51% | **Inconclusive** | zero-shot test requires the unreleased trained ImageNet checkpoints and validation set |
| C5 — cross-modal metrics | **Falsified as written** | the released 85.97-mIoU entrypoint loads ShapeNetPart (16 categories, 50 parts), not ModelNet40; SemanticKITTI code is absent |
| C6 — ablations and cost | **Partial / inconclusive empirics** | paper-table arithmetic checked; no ablation checkpoints/configs; image FLOPs/params rise about 6% with width 396 |

The decisive C5 finding is a benchmark-identity falsification, not a failed
training attempt: the public training and evaluation entrypoints both name
`shapenetcore_partanno_segmentation_benchmark_v0_normal`, instantiate
`PartNormalDataset`, and compute 50-part IoU over 16 ShapeNet categories. The
separate ModelNet40 path is classification and cannot produce the reported
instance-average part-segmentation mIoU.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pytest matplotlib torch
git clone https://github.com/BoyangL1/nD-RoPE vendor/nD-RoPE
git -C vendor/nD-RoPE checkout f2cae70760806451f5e58be4b7e3dc4d0d856a1e
source .venv/bin/activate
PYTHONPATH=repro/src python repro/src/run_claim1.py
PYTHONPATH=repro/src python repro/src/run_claim2.py
PYTHONPATH=repro/src python repro/src/run_source_audit.py
PYTHONPATH=repro/src pytest -q repro/tests
PYTHONPATH=repro/src python repro/src/verify_results.py --root .
```

The executed full configuration is in `repro/configs/full.json`. Raw CSV/JSON
outputs live under `outputs/`. No paper-table value is described as an
independent empirical reproduction unless the underlying experiment was run.

## Hardware and cost

The certificate and audit run on an Apple M2 CPU and require no paid compute.
The paper reports four NVIDIA A100 40 GB GPUs for 400-epoch ImageNet training,
15-epoch Kinetics fine-tuning, and separate 200/50-epoch point-cloud pipelines.
The current HF Jobs four-A100 rate is $10/hour; paid remote compute was not
authorized, and the released repository contains no checkpoints or datasets.
